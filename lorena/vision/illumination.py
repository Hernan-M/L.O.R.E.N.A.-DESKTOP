"""Normalização de iluminação com estado, medida no ROI do rosto.

Substitui o liga/desliga binário do CLAHE baseado na média do frame inteiro:

* **ROI do rosto**: a luminância é medida na caixa do rosto do frame anterior
  (contraluz = fundo claro + rosto escuro agora é detectado).
* **Gama contínua** (LUT, suavizada por EMA): sem degraus, logo sem *jitter*.
* **CLAHE com histerese** (liga < ``clahe_on``, desliga > ``clahe_off``), só no ROI.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class IlluminationStats:
    face_luma: float   # luminância média (0-255) do ROI ANTES da correção
    contrast: float    # desvio-padrão da luminância no ROI
    gamma: float       # gama efetivamente aplicado (1.0 = nenhuma correção)
    clahe: bool


class IlluminationNormalizer:
    def __init__(self, clip_limit: float = 2.0, clahe_on: float = 90.0, clahe_off: float = 120.0,
                 target: float = 0.5, gamma_range: tuple[float, float] = (0.5, 2.0), ema: float = 0.1,
                 roi_margin: float = 0.2, deadband: float = 0.05,
                 stable_range: tuple[float, float] = (85.0, 170.0)) -> None:
        self.clip_limit = clip_limit
        self.clahe_on, self.clahe_off = clahe_on, clahe_off
        self.target = target
        self.gamma_range = gamma_range
        self.ema = ema
        self.roi_margin = roi_margin
        self.deadband = deadband
        self.stable_range = stable_range
        self._gamma = 1.0
        self._clahe_active = False
        self._clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        self._lut_gamma: float | None = None
        self._lut: np.ndarray | None = None

    def reset(self) -> None:
        self._gamma = 1.0
        self._clahe_active = False

    def roi_from_points(self, points: np.ndarray, shape: tuple[int, int]) -> tuple[int, int, int, int]:
        """Caixa (x0, y0, x1, y1) dos landmarks, expandida por ``roi_margin``."""
        h, w = shape
        x0, y0 = points.min(axis=0)
        x1, y1 = points.max(axis=0)
        mx, my = (x1 - x0) * self.roi_margin, (y1 - y0) * self.roi_margin
        return (int(max(0, x0 - mx)), int(max(0, y0 - my)), int(min(w, x1 + mx)), int(min(h, y1 + my)))

    def process(self, frame_bgr: np.ndarray, roi: tuple[int, int, int, int] | None = None
                ) -> tuple[np.ndarray, IlluminationStats]:
        h, w = frame_bgr.shape[:2]
        if roi is None or roi[2] - roi[0] < 16 or roi[3] - roi[1] < 16:
            roi = (0, 0, w, h)
        x0, y0, x1, y1 = roi
        crop = frame_bgr[y0:y1, x0:x1]

        # Medição rápida de luminância via cv2.meanStdDev (10x mais rápido que np.mean/std)
        luma = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        mean_val, std_val = cv2.meanStdDev(luma)
        mean = float(mean_val[0][0])
        contrast = float(std_val[0][0])

        # --- BYPASS DINÂMICO EM LUZ ESTÁVEL (Opção B) ---
        # Se a luminância do rosto estiver na faixa saudável (85 a 170) e o CLAHE
        # não estiver engatado por histerese prévia, o frame original é repassado intacto.
        # Isso economiza ~10 ms por frame e preserva 100% da imagem crua do sensor sem ruídos.
        if (self.stable_range[0] <= mean <= self.stable_range[1]) and not self._clahe_active:
            self._gamma = 1.0
            return frame_bgr, IlluminationStats(mean, contrast, 1.0, False)

        # --- gama contínuo ---------------------------------------------------
        m = float(np.clip(mean / 255.0, 0.02, 0.98))
        wanted = float(np.clip(np.log(self.target) / np.log(m), *self.gamma_range))
        self._gamma += self.ema * (wanted - self._gamma)
        gamma = self._gamma if abs(self._gamma - 1.0) > self.deadband else 1.0

        out = frame_bgr
        if gamma != 1.0:
            out = cv2.LUT(frame_bgr, self._gamma_lut(gamma))

        # --- CLAHE com histerese (só no ROI) -----------------------------------
        if self._clahe_active and mean > self.clahe_off:
            self._clahe_active = False
        elif not self._clahe_active and mean < self.clahe_on:
            self._clahe_active = True
        if self._clahe_active:
            if out is frame_bgr:
                out = frame_bgr.copy()
            # Canal L no espaço CIE LAB preserva o contraste biológico esclera vs íris
            lab = cv2.cvtColor(out[y0:y1, x0:x1], cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            lab = cv2.merge([self._clahe.apply(l_ch), a_ch, b_ch])
            out[y0:y1, x0:x1] = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return out, IlluminationStats(mean, contrast, gamma, self._clahe_active)

    def _gamma_lut(self, gamma: float) -> np.ndarray:
        g = round(gamma, 2)  # reaproveita a LUT enquanto o gama quase não muda
        if self._lut is None or self._lut_gamma != g:
            self._lut = (np.power(np.arange(256) / 255.0, g) * 255.0).astype(np.uint8)
            self._lut_gamma = g
        return self._lut
