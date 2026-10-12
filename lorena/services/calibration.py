"""Sessão de calibração como máquina de estados pura (sem UI, testável).

Para cada ponto: espera rosto → SETTLE (descarta o tempo da sacada até o alvo)
→ COLLECT (coleta ~1 s de amostras válidas) → rejeição de outliers por MAD →
aceita, repete (até ``max_retries``) ou pula o ponto.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from lorena.config import CalibrationSettings
from lorena.domain.models import CalibrationSample, GazeFeatures


class Phase(str, Enum):
    WAIT_FACE = "wait_face"
    SETTLE = "settle"
    COLLECT = "collect"
    DONE = "done"
    FAILED = "failed"


def grid_targets(screen: tuple[int, int], grid: int, margin: int) -> list[tuple[int, int]]:
    w, h = screen
    xs = np.linspace(margin, w - 1 - margin, grid)
    ys = np.linspace(margin, h - 1 - margin, grid)
    # Ordem em linhas (esquerda→direita, cima→baixo): movimento natural do olhar.
    return [(int(x), int(y)) for y in ys for x in xs]


def mad_inliers(values: np.ndarray, k: float = 3.0) -> np.ndarray:
    """Máscara de inliers: |x - mediana| <= k · 1.4826 · MAD (por coluna)."""
    med = np.median(values, axis=0)
    mad = np.median(np.abs(values - med), axis=0) * 1.4826
    mad[mad < 1e-9] = 1e-9
    return np.all(np.abs(values - med) <= k * mad, axis=1)


@dataclass(frozen=True)
class SessionView:
    """Instantâneo para a UI desenhar."""

    phase: Phase
    index: int
    total: int
    target: tuple[int, int] | None
    progress: float
    face_ok: bool
    retry: int


class CalibrationSession:
    def __init__(self, targets: list[tuple[int, int]], settings: CalibrationSettings) -> None:
        if not targets:
            raise ValueError("Sem pontos de calibração.")
        self.targets = targets
        self._s = settings
        self.samples: list[CalibrationSample] = []
        self.skipped: list[int] = []
        self._ears: list[tuple[float, float]] = []
        self._index = 0
        self._retry = 0
        self._phase = Phase.WAIT_FACE
        self._phase_elapsed = 0.0
        self._last_t: float | None = None
        self._buffer: list[GazeFeatures] = []
        self._face_ok = False

    # ------------------------------------------------------------------ state
    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def finished(self) -> bool:
        return self._phase in (Phase.DONE, Phase.FAILED)

    def view(self) -> SessionView:
        if self._phase is Phase.SETTLE:
            progress = min(self._phase_elapsed / self._s.settle_s, 1.0) * 0.4
        elif self._phase is Phase.COLLECT:
            progress = 0.4 + min(self._phase_elapsed / self._s.collect_s, 1.0) * 0.6
        else:
            progress = 0.0
        target = self.targets[self._index] if self._index < len(self.targets) else None
        return SessionView(self._phase, self._index, len(self.targets), target, progress, self._face_ok, self._retry)

    def open_ear_baseline(self) -> tuple[float, float] | None:
        if not self._ears:
            return None
        arr = np.array(self._ears)
        return float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))

    # ----------------------------------------------------------------- update
    def update(self, f: GazeFeatures | None, t: float) -> None:
        if self.finished:
            return
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t
        valid = f is not None and f.quality >= self._s.min_quality and math.isfinite(f.h) and math.isfinite(f.v)
        self._face_ok = valid

        if self._phase is Phase.WAIT_FACE:
            if valid:
                self._set_phase(Phase.SETTLE)
            return

        if not valid:
            # Pausa a contagem; se o rosto sumiu no meio da coleta, recomeça o ponto.
            if f is None:
                self._buffer.clear()
                self._set_phase(Phase.WAIT_FACE)
            return

        self._phase_elapsed += dt
        if self._phase is Phase.SETTLE:
            if self._phase_elapsed >= self._s.settle_s:
                self._set_phase(Phase.COLLECT)
        elif self._phase is Phase.COLLECT:
            self._buffer.append(f)
            if self._phase_elapsed >= self._s.collect_s:
                self._finish_point()

    def _set_phase(self, phase: Phase) -> None:
        self._phase = phase
        self._phase_elapsed = 0.0
        if phase is Phase.COLLECT:
            self._buffer = []

    def _finish_point(self) -> None:
        buf = self._buffer
        accepted = False
        if len(buf) >= self._s.min_samples:
            hv = np.array([(f.h, f.v) for f in buf])
            mask = mad_inliers(hv)
            inliers = [f for f, ok in zip(buf, mask) if ok]
            if len(inliers) >= self._s.min_samples:
                spread = np.median(np.abs(hv[mask] - np.median(hv[mask], axis=0)), axis=0) * 1.4826
                accepted = bool(np.all(spread <= self._s.max_spread))
            if accepted:
                target = self.targets[self._index]
                self.samples.extend(CalibrationSample.from_features(f, target, self._index) for f in inliers)
                self._ears.extend((f.ear_left, f.ear_right) for f in inliers)

        if not accepted and self._retry < self._s.max_retries:
            self._retry += 1
            self._set_phase(Phase.SETTLE)
            return
        if not accepted:
            self.skipped.append(self._index)
        self._index += 1
        self._retry = 0
        if self._index >= len(self.targets):
            needed = 6 if self._s.polynomial_degree == 2 else 3
            self._phase = Phase.DONE if len(self.targets) - len(self.skipped) >= needed else Phase.FAILED
        else:
            self._set_phase(Phase.SETTLE)
