"""Filtros de estabilização do cursor.

Ordem no pipeline: OutlierGate → OneEuroFilter2D → FixationStabilizer.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np


class OneEuroFilter:
    """One Euro Filter (Casiez, Roussel & Vogel, CHI 2012).

    Corte adaptativo: f_c = min_cutoff + beta·|dx/dt|. Suaviza muito quando o
    sinal está parado (tremor) e pouco quando se move rápido (baixa latência).
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0) -> None:
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self) -> None:
        self._x: float | None = None
        self._dx = 0.0
        self._t: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t: float) -> float:
        if self._x is None or self._t is None:
            self._x, self._t = x, t
            return x
        dt = max(t - self._t, 1e-3)
        dx = (x - self._x) / dt
        self._dx = self._dx + self._alpha(self.d_cutoff, dt) * (dx - self._dx)
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        self._x = self._x + self._alpha(cutoff, dt) * (x - self._x)
        self._t = t
        return self._x


    def set_min_cutoff(self, min_cutoff: float) -> None:
        self.min_cutoff = min_cutoff


class OneEuroFilter2D:
    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        self._fx = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self._fy = OneEuroFilter(min_cutoff, beta, d_cutoff)

    def set_min_cutoff(self, min_cutoff: float) -> None:
        self._fx.set_min_cutoff(min_cutoff)
        self._fy.set_min_cutoff(min_cutoff)

    def reset(self) -> None:
        self._fx.reset()
        self._fy.reset()

    def __call__(self, p: np.ndarray, t: float) -> np.ndarray:
        return np.array([self._fx(float(p[0]), t), self._fy(float(p[1]), t)])


class OutlierGate:
    """Segura saltos grandes até que se confirmem por ``confirm_frames`` frames.

    Uma sacada real se mantém no novo local; um erro de detecção é isolado.
    Aceita deslocamento conhecido (``shift``, p. ex. por pose da cabeça).
    """

    def __init__(self, jump_px: float, confirm_frames: int) -> None:
        self.jump_px, self.confirm_frames = jump_px, confirm_frames
        self.reset()

    def reset(self) -> None:
        self._ref: np.ndarray | None = None
        self._pending: list[np.ndarray] = []

    def apply_shift(self, shift: np.ndarray) -> None:
        """Aplica deslocamento físico de cabeça na referência sem resetar o gate."""
        if self._ref is not None:
            self._ref = self._ref + shift

    def __call__(self, p: np.ndarray, shift: np.ndarray | None = None) -> np.ndarray:
        if self._ref is None:
            self._ref = p.copy()
            return p
        ref_cand = self._ref + shift if shift is not None else self._ref
        if np.linalg.norm(p - ref_cand) <= self.jump_px:
            self._pending.clear()
            self._ref = p.copy()
            return p
        if self._pending and np.linalg.norm(p - self._pending[0]) > self.jump_px:
            self._pending = []  # candidatos inconsistentes entre si: recomeça
        self._pending.append(p.copy())
        if len(self._pending) >= self.confirm_frames:
            if len(self._pending) >= 2:
                self._ref = 0.3 * self._pending[-2] + 0.7 * self._pending[-1]
            else:
                self._ref = self._pending[-1].copy()
            self._pending.clear()
        return self._ref


class FixationStabilizer:
    """Zona morta contínua C^0: núcleo estrito (0.5 R) + anel de transição quadrática suave.

    Absorve micro-sacadas sem degrau visual e mantém a âncora de fixação para clique por permanência.
    """

    def __init__(self, radius_px: float, inner_ratio: float = 0.5) -> None:
        self.radius_px = radius_px
        self.inner_ratio = inner_ratio
        self.reset()

    def reset(self) -> None:
        self.center: np.ndarray | None = None
        self._anchor: np.ndarray | None = None
        self.fixation_start: float = 0.0

    def __call__(self, p: np.ndarray, t: float) -> np.ndarray:
        if self.center is None or self._anchor is None:
            self.center = p.copy()
            self._anchor = p.copy()
            self.fixation_start = t
            return self.center

        r = self.radius_px
        r_in = self.inner_ratio * r
        d = float(np.linalg.norm(p - self.center))

        if d <= r_in or d < 1e-9:
            c_new = self.center
        elif d <= r:
            u = (p - self.center) / d
            s = (d - r_in) / max(r - r_in, 1e-6)
            c_new = self.center + (s * s * (d - r_in)) * u
        else:
            u = (p - self.center) / d
            c_new = p - r_in * u

        # Ancoragem do dwell timer enquanto permanecer no raio de fixação da âncora
        dist_from_anchor = float(np.linalg.norm(c_new - self._anchor))
        if dist_from_anchor <= r:
            self.center = c_new
        else:
            self.center = c_new
            self._anchor = c_new.copy()
            self.fixation_start = t

        return self.center

    def fixation_duration(self, t: float) -> float:
        return 0.0 if self.center is None else max(0.0, t - self.fixation_start)


class DwellClicker:
    """Clique por permanência: dispara uma vez por fixação após ``dwell_s``."""

    def __init__(self, dwell_s: float) -> None:
        self.dwell_s = dwell_s
        self._fired_for: float | None = None

    def update(self, stabilizer: FixationStabilizer, t: float) -> tuple[bool, float]:
        duration = stabilizer.fixation_duration(t)
        progress = min(duration / self.dwell_s, 1.0) if self.dwell_s > 0 else 0.0
        if self._fired_for == stabilizer.fixation_start:
            return False, 0.0
        if progress >= 1.0:
            self._fired_for = stabilizer.fixation_start
            return True, 1.0
        return False, progress


class PositionHistory:
    """Buffer circular (t, p) para recuperar a posição de antes de um gesto."""

    def __init__(self, maxlen: int = 90) -> None:
        self._buf: deque[tuple[float, np.ndarray]] = deque(maxlen=maxlen)

    def push(self, t: float, p: np.ndarray) -> None:
        self._buf.append((t, p))

    def at(self, t: float) -> np.ndarray | None:
        for ts, p in reversed(self._buf):
            if ts <= t:
                return p
        return self._buf[0][1] if self._buf else None

    def clear(self) -> None:
        self._buf.clear()
