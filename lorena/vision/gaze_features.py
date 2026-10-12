"""Extração das características do olhar (h, v) a partir da geometria dos olhos.

Para cada olho, com cantos C_s→C_e (esquerda→direita da imagem) e íris I:

    u = (C_e - C_s)/W,  n = perp(u),  W = |C_e - C_s|,  M = (C_s + C_e)/2
    h = (I - C_s)·u / W        v = (I - M)·n / W

* Float/subpixel (sem quantização de pixel inteiro).
* Vertical normalizada pela LARGURA do olho (estável), não pela altura (que
  muda com a pálpebra).
* Projeção no eixo do olho compensa roll e distância da câmera.
* Os dois olhos são combinados com peso proporcional à abertura: se um falha
  ou fecha, o outro assume.
"""
from __future__ import annotations

import numpy as np

from lorena.config import BlinkSettings
from lorena.domain.models import EyeObservation, FaceObservation, GazeFeatures


def eye_ratios(iris: np.ndarray, start: np.ndarray, end: np.ndarray) -> tuple[float, float] | None:
    axis = end - start
    width = float(np.linalg.norm(axis))
    if width < 1e-6:
        return None
    u = axis / width
    n = np.array([-u[1], u[0]])
    h = float(np.dot(iris - start, u) / width)
    v = float(np.dot(iris - (start + end) / 2.0, n) / width)
    return h, v


class OpennessModel:
    """Converte EAR bruto em abertura normalizada [0, 1] por olho e por usuário."""

    def __init__(self, settings: BlinkSettings, open_left: float | None = None, open_right: float | None = None) -> None:
        self._s = settings
        self.open_left = open_left or settings.default_open_ear
        self.open_right = open_right or settings.default_open_ear

    def set_baseline(self, open_left: float, open_right: float) -> None:
        self.open_left, self.open_right = open_left, open_right

    def openness(self, ear: float, open_ear: float) -> float:
        closed = open_ear * self._s.closed_ratio
        span = max(open_ear - closed, 1e-6)
        return float(np.clip((ear - closed) / span, 0.0, 1.0))


class GazeFeatureExtractor:
    def __init__(self, openness: OpennessModel) -> None:
        self.openness = openness

    def extract(self, face: FaceObservation | None) -> GazeFeatures | None:
        if face is None:
            return None
        o_left = self.openness.openness(face.left_eye.ear, self.openness.open_left)
        o_right = self.openness.openness(face.right_eye.ear, self.openness.open_right)

        acc_h = acc_v = acc_w = 0.0
        for eye, weight in ((face.left_eye, o_left), (face.right_eye, o_right)):
            ratios = self._ratios(eye)
            if ratios is None or weight <= 0.0:
                continue
            acc_h += weight * ratios[0]
            acc_v += weight * ratios[1]
            acc_w += weight
        if acc_w <= 1e-6:
            # Olhos fechados: ainda devolvemos aberturas (necessárias para a piscada).
            return GazeFeatures(face.timestamp, float("nan"), float("nan"), face.yaw, face.pitch,
                                face.left_eye.ear, face.right_eye.ear, o_left, o_right, 0.0)

        quality = float(np.clip(max(o_left, o_right) * 0.5 + min(o_left, o_right) * 0.5, 0.0, 1.0))
        return GazeFeatures(
            timestamp=face.timestamp,
            h=acc_h / acc_w,
            v=acc_v / acc_w,
            yaw=face.yaw,
            pitch=face.pitch,
            ear_left=face.left_eye.ear,
            ear_right=face.right_eye.ear,
            openness_left=o_left,
            openness_right=o_right,
            quality=quality,
        )

    @staticmethod
    def _ratios(eye: EyeObservation) -> tuple[float, float] | None:
        return eye_ratios(eye.iris, eye.start_corner, eye.end_corner)
