"""Controle de exposição da webcam em malha fechada, medido no rosto.

Estratégia (DirectShow: ``CAP_PROP_EXPOSURE`` em log2 segundos; 1 passo = 1 stop):

1. Tenta modo manual (``CAP_PROP_AUTO_EXPOSURE`` = 0,25) e **confere** que o driver
   aceitou (lê de volta). Se não aceitou: restaura o automático (``active = False``) e
   só a normalização por software (gama/CLAHE) atua.
2. Controlador com banda morta: luminância alvo do rosto ± ``deadband``; no máximo 1
   ajuste de ±1 stop a cada ``min_interval`` s (sem *flicker* que confunda o MediaPipe).
3. Limita a exposição para manter ≥ ``fps_floor`` FPS (exposição longa = FPS baixo e
   borrão quando a cabeça se mexe). Se não bastar, sobe o ganho (limitado).
4. ``restore()`` devolve as configurações originais da câmera.
"""
from __future__ import annotations

import logging
import math
from typing import Protocol

import cv2

log = logging.getLogger(__name__)

MANUAL_MODE = 0.25
AUTO_MODE = 0.75
MIN_EXPOSURE_LOG2 = -11.0


class _Capture(Protocol):
    def get(self, prop: int) -> float: ...
    def set(self, prop: int, value: float) -> bool: ...


class ExposureController:
    def __init__(self, cap: _Capture, target_luma: float = 120.0, deadband: float = 15.0, min_interval: float = 0.5,
                 fps_floor: float = 25.0, gain_headroom: float = 32.0, gain_step: float = 4.0) -> None:
        self._cap = cap
        self.target_luma = target_luma
        self.deadband = deadband
        self.min_interval = min_interval
        self.max_exposure = float(math.floor(-math.log2(max(fps_floor, 1.0))))  # 2^e <= 1/fps
        self.min_exposure = MIN_EXPOSURE_LOG2
        self.gain_headroom, self.gain_step = gain_headroom, gain_step
        self.active = False
        self._last_adjust = -1e9
        self._orig_auto = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        self._orig_exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
        self._orig_gain = cap.get(cv2.CAP_PROP_GAIN)
        self._exposure = self._orig_exposure
        self._gain = self._orig_gain

    def start(self) -> bool:
        """Tenta assumir o controle manual. Retorna False (e restaura) se o driver rejeitar."""
        try:
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, MANUAL_MODE)
            start = min(max(self._orig_exposure, self.min_exposure), self.max_exposure)
            ok = self._cap.set(cv2.CAP_PROP_EXPOSURE, start)
            readback = self._cap.get(cv2.CAP_PROP_EXPOSURE)
            mode = self._cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
            accepted = bool(ok) and math.isfinite(readback) and abs(readback - start) <= 0.5 and abs(mode - MANUAL_MODE) < 0.1
        except Exception:  # noqa: BLE001 - drivers imprevisíveis
            accepted = False
        if not accepted:
            log.info("Driver da câmera não aceitou exposição manual; usando só correção por software.")
            self.restore()
            self.active = False
            return False
        self._exposure = start
        self.active = True
        log.info("Controle de exposição ativo (alvo de luminância %.0f, exposição inicial %.0f).", self.target_luma, start)
        return True

    def update(self, face_luma: float, now: float) -> None:
        if not self.active or face_luma <= 0 or now - self._last_adjust < self.min_interval:
            return
        err = self.target_luma - face_luma
        if abs(err) <= self.deadband:
            return
        step = 1.0 if err > 0 else -1.0
        self._last_adjust = now
        new_exp = min(max(self._exposure + step, self.min_exposure), self.max_exposure)
        if new_exp != self._exposure:
            ok = self._cap.set(cv2.CAP_PROP_EXPOSURE, new_exp)
            readback = self._cap.get(cv2.CAP_PROP_EXPOSURE)
            if ok and math.isfinite(readback) and abs(readback - new_exp) <= 0.5:
                self._exposure = new_exp
            else:
                log.info("Driver não alterou exposição de hardware; desativando malha para prevenir flicker.")
                self.restore()
                self.active = False
            return
        # Exposição no limite: usa o ganho (limitado) como complemento.
        if math.isfinite(self._orig_gain) and self._orig_gain >= 0:
            lo, hi = self._orig_gain, self._orig_gain + self.gain_headroom
            new_gain = min(max(self._gain + step * self.gain_step, lo), hi)
            if new_gain != self._gain and self._cap.set(cv2.CAP_PROP_GAIN, new_gain):
                self._gain = new_gain

    def restore(self) -> None:
        """Devolve exposição/ganho/modo automático originais."""
        try:
            if math.isfinite(self._orig_exposure):
                self._cap.set(cv2.CAP_PROP_EXPOSURE, self._orig_exposure)
            if math.isfinite(self._orig_gain) and self._orig_gain >= 0:
                self._cap.set(cv2.CAP_PROP_GAIN, self._orig_gain)
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, self._orig_auto if self._orig_auto > 0 else AUTO_MODE)
        except Exception:  # noqa: BLE001
            log.debug("Falha ao restaurar a exposição da câmera.", exc_info=True)
        self.active = False
