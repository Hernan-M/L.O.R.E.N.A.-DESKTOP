"""Extração das características do olhar (h, v) a partir da geometria dos olhos.

Para cada olho, com cantos C_s→C_e (esquerda→direita da imagem) e íris I:

    u = (C_e - C_s)/W,  n = perp(u),  W = |C_e - C_s|,  M = (C_s + C_e)/2
    h = (I - C_s)·u / W        v = (I - M)·n / W

* Float/subpixel (sem quantização de pixel inteiro).
* Vertical normalizada pela LARGURA do olho (estável), não pela altura (que
  muda com a pálpebra).
* Projeção no eixo do olho compensa roll e distância da câmera.
* Os dois olhos são combinados com peso proporcional à abertura × largura²: se um falha
  ou fecha, o outro assume, e o olho mais distante/encurtado (mais ruidoso) pesa menos.
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


def eye_lid(upper_lid: np.ndarray, start: np.ndarray, end: np.ndarray) -> float | None:
    """Posição vertical (assinada) da pálpebra superior em relação ao eixo do olho, / largura.

    Acompanha o olhar para baixo (a pálpebra desce junto) e compensa a oclusão parcial da
    íris, que faz o landmark da íris ser "puxado" para cima nos cantos inferiores da tela.
    """
    axis = end - start
    width = float(np.linalg.norm(axis))
    if width < 1e-6:
        return None
    u = axis / width
    n = np.array([-u[1], u[0]])
    return float(np.dot(upper_lid - (start + end) / 2.0, n) / width)

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
    MIN_EYE_PX = 18.0  # olho mais estreito que isso (em px) reduz a qualidade

    def __init__(self, openness: OpennessModel) -> None:
        self.openness = openness
        self._disp_h: float | None = None
        self._disp_v: float | None = None
        self._disp_lid: float | None = None
        self._disp_alpha: float = 0.05

    def reset(self) -> None:
        """Reinicia os estimadores de disparidade binocular."""
        self._disp_h = None
        self._disp_v = None
        self._disp_lid = None

    @staticmethod
    def _eye_weight(o: float) -> float:
        """Função de peso com platô constante (50%/50%) para abertura >= 0.70 e rampa suave."""
        if o >= 0.70:
            return 1.0
        return float(np.clip((o - 0.15) / 0.55, 0.0, 1.0)) ** 2

    def extract(self, face: FaceObservation | None) -> GazeFeatures | None:
        if face is None:
            return None
        o_left = self.openness.openness(face.left_eye.ear, self.openness.open_left)
        o_right = self.openness.openness(face.right_eye.ear, self.openness.open_right)

        ratios_l = self._ratios(face.left_eye)
        ratios_r = self._ratios(face.right_eye)

        w_left_px = float(np.linalg.norm(face.left_eye.end_corner - face.left_eye.start_corner))
        w_right_px = float(np.linalg.norm(face.right_eye.end_corner - face.right_eye.start_corner))

        lid_l = eye_lid(face.left_eye.upper_lid, face.left_eye.start_corner, face.left_eye.end_corner) if face.left_eye.upper_lid is not None else None
        lid_r = eye_lid(face.right_eye.upper_lid, face.right_eye.start_corner, face.right_eye.end_corner) if face.right_eye.upper_lid is not None else None

        # Atualiza disparidade binocular via EMA lenta (alpha = 0.05) quando ambos os olhos estão bem abertos (>= 0.75)
        if ratios_l is not None and ratios_r is not None and min(o_left, o_right) >= 0.75:
            dh = ratios_l[0] - ratios_r[0]
            dv = ratios_l[1] - ratios_r[1]
            if self._disp_h is None or self._disp_v is None:
                self._disp_h, self._disp_v = dh, dv
            else:
                self._disp_h = (1.0 - self._disp_alpha) * self._disp_h + self._disp_alpha * dh
                self._disp_v = (1.0 - self._disp_alpha) * self._disp_v + self._disp_alpha * dv

            if lid_l is not None and lid_r is not None:
                dlid = lid_l - lid_r
                if self._disp_lid is None:
                    self._disp_lid = dlid
                else:
                    self._disp_lid = (1.0 - self._disp_alpha) * self._disp_lid + self._disp_alpha * dlid

        # Compensação de disparidade/paralaxe em relação ao referencial ciclópico médio
        dh_half = (self._disp_h * 0.5) if self._disp_h is not None else 0.0
        dv_half = (self._disp_v * 0.5) if self._disp_v is not None else 0.0
        dlid_half = (self._disp_lid * 0.5) if self._disp_lid is not None else 0.0

        w_l = self._eye_weight(o_left) if ratios_l is not None else 0.0
        w_r = self._eye_weight(o_right) if ratios_r is not None else 0.0

        acc_o = o_left + o_right
        common = dict(yaw=face.yaw, pitch=face.pitch, head_x=face.head_x, head_y=face.head_y,
                      head_z=face.head_z, face_luma=face.luma)

        if w_l + w_r <= 1e-6 or acc_o <= 1e-6:
            # Olhos fechados ou sem detecção válida: devolve aberturas para máquina de estados de piscadas
            return GazeFeatures(face.timestamp, float("nan"), float("nan"), face.yaw, face.pitch,
                                face.left_eye.ear, face.right_eye.ear, o_left, o_right, 0.0,
                                head_x=face.head_x, head_y=face.head_y, head_z=face.head_z, face_luma=face.luma)

        hl_corr = (ratios_l[0] - dh_half) if ratios_l is not None else 0.0
        vl_corr = (ratios_l[1] - dv_half) if ratios_l is not None else 0.0
        hr_corr = (ratios_r[0] + dh_half) if ratios_r is not None else 0.0
        vr_corr = (ratios_r[1] + dv_half) if ratios_r is not None else 0.0

        h_fused = (w_l * hl_corr + w_r * hr_corr) / (w_l + w_r)
        v_fused = (w_l * vl_corr + w_r * vr_corr) / (w_l + w_r)

        # Fusão de pálpebra com compensação
        lid_fused = 0.0
        w_lid_tot = 0.0
        if lid_l is not None and w_l > 0.0:
            lid_fused += w_l * (lid_l - dlid_half)
            w_lid_tot += w_l
        if lid_r is not None and w_r > 0.0:
            lid_fused += w_r * (lid_r + dlid_half)
            w_lid_tot += w_r
        if w_lid_tot > 0.0:
            lid_fused /= w_lid_tot

        acc_wpx = o_left * w_left_px + o_right * w_right_px
        eye_px = acc_wpx / acc_o
        size_factor = float(np.clip(eye_px / self.MIN_EYE_PX, 0.0, 1.0))
        quality = float(np.clip(max(o_left, o_right) * 0.5 + min(o_left, o_right) * 0.5, 0.0, 1.0)) * size_factor

        return GazeFeatures(
            timestamp=face.timestamp,
            h=h_fused,
            v=v_fused,
            ear_left=face.left_eye.ear,
            ear_right=face.right_eye.ear,
            openness_left=o_left,
            openness_right=o_right,
            quality=quality,
            lid=lid_fused,
            eye_px=eye_px,
            **common,
        )

    @staticmethod
    def _ratios(eye: EyeObservation) -> tuple[float, float] | None:
        return eye_ratios(eye.iris, eye.start_corner, eye.end_corner)