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
    """One Euro Filter 2D adaptativo para suavização em 30 FPS.

    Em modo isotrópico (padrão), utiliza a norma Euclidiana da velocidade 2D
    v = ||v||_2 para calcular uma frequência de corte única compartilhada entre
    os eixos X e Y. Isso garante invariância rotacional, prevenindo distorção
    de trajetória (curvamento) em movimentos diagonais e oblíquos.
    Zero alocações desnecessárias no hot loop a 30 FPS.
    """

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float, isotropic: bool = True) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.isotropic = isotropic
        self._fx = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self._fy = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self._x = np.zeros(2, dtype=float)
        self._dx = np.zeros(2, dtype=float)
        self._t: float | None = None
        self._initialized = False

    def set_min_cutoff(self, min_cutoff: float) -> None:
        self.min_cutoff = min_cutoff
        self._fx.set_min_cutoff(min_cutoff)
        self._fy.set_min_cutoff(min_cutoff)

    def reset(self) -> None:
        self._fx.reset()
        self._fy.reset()
        self._initialized = False
        self._t = None
        self._dx.fill(0.0)

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, p: np.ndarray, t: float, out: np.ndarray | None = None) -> np.ndarray:
        px = float(p[0])
        py = float(p[1])

        if not self._initialized or self._t is None:
            self._x[0] = px
            self._x[1] = py
            self._dx.fill(0.0)
            self._t = t
            self._initialized = True
            self._fx.reset()
            self._fy.reset()
            self._fx(px, t)
            self._fy(py, t)
            if out is not None:
                out[0] = px
                out[1] = py
                return out
            return self._x.copy()

        if not self.isotropic:
            res_x = self._fx(px, t)
            res_y = self._fy(py, t)
            self._x[0] = res_x
            self._x[1] = res_y
            self._t = t
            if out is not None:
                out[0] = res_x
                out[1] = res_y
                return out
            return np.array([res_x, res_y], dtype=float)

        # inv: dt >= 1e-3
        dt = max(t - self._t, 1e-3)
        raw_dx = (px - self._x[0]) / dt
        raw_dy = (py - self._x[1]) / dt

        alpha_d = self._alpha(self.d_cutoff, dt)
        self._dx[0] += alpha_d * (raw_dx - self._dx[0])
        self._dx[1] += alpha_d * (raw_dy - self._dx[1])

        # Norma Euclidiana invariante a rotações
        speed = math.hypot(self._dx[0], self._dx[1])

        cutoff = self.min_cutoff + self.beta * speed
        alpha = self._alpha(cutoff, dt)

        self._x[0] += alpha * (px - self._x[0])
        self._x[1] += alpha * (py - self._x[1])
        self._t = t

        if out is not None:
            out[0] = self._x[0]
            out[1] = self._x[1]
            return out
        return self._x.copy()


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

    def __call__(self, p: np.ndarray, shift: np.ndarray | None = None) -> np.ndarray:
        if self._ref is None:
            self._ref = np.asarray(p, dtype=float).copy()
            return self._ref
        if shift is not None:
            self._ref[0] += float(shift[0])
            self._ref[1] += float(shift[1])

        dx = float(p[0] - self._ref[0])
        dy = float(p[1] - self._ref[1])
        dist = math.hypot(dx, dy)

        if dist <= self.jump_px:
            self._pending.clear()
            self._ref[0] = float(p[0])
            self._ref[1] = float(p[1])
            return self._ref

        if self._pending:
            p0 = self._pending[0]
            d0 = math.hypot(float(p[0] - p0[0]), float(p[1] - p0[1]))
            if d0 > self.jump_px:
                self._pending.clear()  # candidatos inconsistentes entre si: recomeça

        self._pending.append(np.asarray(p, dtype=float).copy())
        if len(self._pending) >= self.confirm_frames:
            mean_x = sum(float(c[0]) for c in self._pending) / len(self._pending)
            mean_y = sum(float(c[1]) for c in self._pending) / len(self._pending)
            self._ref[0] = mean_x
            self._ref[1] = mean_y
            self._pending.clear()
        return self._ref


class FixationStabilizer:
    """Zona morta e estabilização de fixação adaptativa.

    Absorve micro-sacadas fisiológicas e ruído de rastreio enquanto o olhar
    estiver dentro de ``radius_px``. Com ``smooth=True``, implementa uma âncora
    contínua com taxa de convergência (catch-up) que elimina o efeito escada (ratchet)
    ao seguir alvos em baixa velocidade E garante convergência assintótica suave
    ao alvo de fixação final (sem erro residual de arrasto), enquanto sacadas amplas
    (> break_mult * radius) rompem a âncora imediatamente para novo foco.
    """

    def __init__(
        self,
        radius_px: float,
        smooth: bool = True,
        break_mult: float = 2.5,
        catchup_rate: float = 0.25,
    ) -> None:
        self.radius_px = radius_px
        self.smooth = smooth
        self.break_mult = break_mult
        self.catchup_rate = catchup_rate
        self.reset()

    def reset(self) -> None:
        self.center: np.ndarray | None = None
        self.fixation_start: float = 0.0
        self._dragging: bool = False
        self._last_p: tuple[float, float] | None = None
        self._last_t: float | None = None

    def __call__(self, p: np.ndarray, t: float) -> np.ndarray:
        px = float(p[0])
        py = float(p[1])
        if self.center is None or self._last_p is None or self._last_t is None:
            self.center = np.array([px, py], dtype=float)
            self.fixation_start = t
            self._dragging = False
            self._last_p = (px, py)
            self._last_t = t
            return self.center.copy()

        dt = max(t - self._last_t, 1e-3)
        gaze_speed = math.hypot(px - self._last_p[0], py - self._last_p[1]) / dt
        self._last_p = (px, py)
        self._last_t = t

        dx = px - self.center[0]
        dy = py - self.center[1]
        dist = math.hypot(dx, dy)

        if not self.smooth:
            if dist > self.radius_px:
                self.center[0] = px
                self.center[1] = py
                self.fixation_start = t
            return self.center.copy()

        break_radius = self.radius_px * self.break_mult
        if dist >= break_radius:
            # Sacada rápida ampla: salta para o novo alvo e reinicia a fixação
            self.center[0] = px
            self.center[1] = py
            self.fixation_start = t
            self._dragging = False
            return self.center.copy()

        # Quando o olhar para ou desacelera (repouso no alvo final), converge a âncora:
        if gaze_speed < 40.0:
            if dist <= 1.0:
                self.center[0] = px
                self.center[1] = py
                self._dragging = False
            else:
                self.center[0] += dx * self.catchup_rate
                self.center[1] += dy * self.catchup_rate
            return self.center.copy()

        if dist > self.radius_px:
            # Movimento contínuo suave (leaky anchor C^0 contínua):
            # Desloca a âncora suavemente na direção do olhar sem degrau
            excess = dist - self.radius_px
            scale = excess / dist
            self.center[0] += dx * scale
            self.center[1] += dy * scale
            self._dragging = True
            self.fixation_start = t
            return self.center.copy()

        # Fixação estável dentro da zona morta: absorve todo tremor/ruído
        return self.center.copy()

    def fixation_duration(self, t: float) -> float:
        return 0.0 if self.center is None else t - self.fixation_start


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
        self._buf.append((t, np.array([float(p[0]), float(p[1])], dtype=float)))

    def at(self, t: float) -> np.ndarray | None:
        for ts, p in reversed(self._buf):
            if ts <= t:
                return p.copy()
        return self._buf[0][1].copy() if self._buf else None

    def clear(self) -> None:
        self._buf.clear()
