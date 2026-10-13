"""Sessão de calibração como máquina de estados pura (sem UI, testável).

Para cada ponto: espera rosto → SETTLE (descarta o tempo da sacada até o alvo)
→ COLLECT (coleta ~1 s de amostras válidas) → rejeição de outliers por MAD →
aceita, repete (até ``max_retries``) ou pula o ponto.

Calibração em 2 passes (público que mexe muito a cabeça):

* **Passe 1** (grade 3×3): cabeça confortável; ajusta o modelo "olho na cabeça".
* **Passe 2** (5 pontos, ``pose_start``): o usuário vira levemente a cabeça; o ponto só
  entra em COLLECT quando o giro passa de ``pose_min_deg`` (ou após ``pose_timeout_s``,
  aceitando o maior giro obtido). Isso gera variação de pose para identificar o efeito
  da cabeça, que antes (σ_yaw ≈ 0,5°) era ruído ajustado pelo modelo.
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


def resolve_margin(screen: tuple[int, int], cfg: CalibrationSettings) -> int:
    """Margem dos alvos: relativa ao lado menor da tela (mín. 24 px) ou, se 0, ``margin_px``."""
    if cfg.margin_frac > 0:
        return max(24, int(cfg.margin_frac * min(screen)))
    return cfg.margin_px


POSE_HINTS = ["esquerda", "cima e à esquerda", "cima e à direita", "baixo e à esquerda", "baixo e à direita"]


def calibration_targets(screen: tuple[int, int], cfg: CalibrationSettings
                        ) -> tuple[list[tuple[int, int]], int | None, list[str]]:
    """(alvos, índice do 1º ponto do passe de pose ou None, dicas de direção do passe 2)."""
    margin = resolve_margin(screen, cfg)
    points = grid_targets(screen, cfg.grid, margin)
    if not (cfg.pose_pass and cfg.use_head_pose):
        return points, None, []
    w, h = screen
    pose_points = [(w // 2, h // 2), (margin, margin), (w - 1 - margin, margin),
                   (margin, h - 1 - margin), (w - 1 - margin, h - 1 - margin)]
    return points + pose_points, len(points), list(POSE_HINTS)


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
    pose_hint: str | None = None      # direção para virar a cabeça (passe 2)
    pose_dev_deg: float = 0.0         # giro atual em relação à pose de referência
    pose_min_deg: float = 0.0         # giro mínimo exigido


class CalibrationSession:
    SPREAD_FLOOR = 0.008  # piso do limite de dispersão relativo (unidades de h/v)

    def __init__(self, targets: list[tuple[int, int]], settings: CalibrationSettings,
                 pose_start: int | None = None, pose_hints: list[str] | None = None) -> None:
        if not targets:
            raise ValueError("Sem pontos de calibração.")
        self.targets = targets
        self._s = settings
        self.pose_start = pose_start
        self._pose_hints = pose_hints or []
        self.samples: list[CalibrationSample] = []
        self.skipped: list[int] = []
        self.pose_ref: tuple[float, float, float, float] | None = None  # (yaw, pitch, hx, hy) do passe 1
        self._ears: list[tuple[float, float]] = []
        self._lumas: list[float] = []
        self._medians: list[np.ndarray] = []
        self._index = 0
        self._retry = 0
        self._phase = Phase.WAIT_FACE
        self._phase_elapsed = 0.0
        self._pose_wait = 0.0
        self._pose_dev = 0.0
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

    def _is_pose_point(self) -> bool:
        return self.pose_start is not None and self._index >= self.pose_start and self.pose_ref is not None

    def view(self) -> SessionView:
        if self._phase is Phase.SETTLE:
            progress = min(self._phase_elapsed / self._s.settle_s, 1.0) * 0.4
        elif self._phase is Phase.COLLECT:
            progress = 0.4 + min(self._phase_elapsed / self._s.collect_s, 1.0) * 0.6
        else:
            progress = 0.0
        target = self.targets[self._index] if self._index < len(self.targets) else None
        hint = None
        if self._is_pose_point() and not self.finished:
            k = self._index - self.pose_start
            hint = self._pose_hints[k] if k < len(self._pose_hints) else "para o lado"
        return SessionView(self._phase, self._index, len(self.targets), target, progress, self._face_ok, self._retry,
                           hint, self._pose_dev, self._s.pose_min_deg if hint else 0.0)

    def open_ear_baseline(self) -> tuple[float, float] | None:
        if not self._ears:
            return None
        arr = np.array(self._ears)
        return float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))

    def face_luma_baseline(self) -> float | None:
        """Luminância mediana do rosto durante a calibração (para detectar mudança de luz depois)."""
        lumas = [x for x in self._lumas if x > 0]
        return float(np.median(lumas)) if lumas else None

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

        pose_ok = True
        if self._is_pose_point():
            self._pose_wait += dt
            self._pose_dev = self._deviation_deg(f)
            timed_out = self._pose_wait >= self._s.pose_timeout_s
            need = self._s.pose_min_deg if self._phase is Phase.SETTLE else 0.6 * self._s.pose_min_deg
            pose_ok = self._pose_dev >= need or timed_out
        else:
            self._pose_dev = 0.0

        if not pose_ok:
            return  # aguarda o usuário virar a cabeça (não avança o relógio da fase)

        self._phase_elapsed += dt
        if self._phase is Phase.SETTLE:
            if self._phase_elapsed >= self._s.settle_s:
                self._set_phase(Phase.COLLECT)
        elif self._phase is Phase.COLLECT:
            self._buffer.append(f)
            if self._phase_elapsed >= self._s.collect_s:
                self._finish_point()

    def _deviation_deg(self, f: GazeFeatures) -> float:
        yaw0, pitch0 = self.pose_ref[0], self.pose_ref[1]
        return math.degrees(math.hypot(f.yaw - yaw0, f.pitch - pitch0))

    def _set_phase(self, phase: Phase) -> None:
        self._phase = phase
        self._phase_elapsed = 0.0
        if phase is Phase.COLLECT:
            self._buffer = []

    def _spread_limit(self) -> np.ndarray:
        """Limite de dispersão por eixo: relativo à faixa de h/v já observada (telas pequenas têm faixa menor)."""
        limit = np.full(2, self._s.max_spread)
        if len(self._medians) >= 4:
            rng = np.ptp(np.array(self._medians), axis=0)
            limit = np.clip(self._s.max_spread_frac * rng, self.SPREAD_FLOOR, self._s.max_spread)
        return limit

    def _compute_pose_ref(self) -> None:
        if self.samples:
            a = np.array([(s.yaw, s.pitch, s.head_x, s.head_y) for s in self.samples])
            self.pose_ref = tuple(float(x) for x in np.median(a, axis=0))

    def _finish_point(self) -> None:
        buf = self._buffer
        accepted = False
        if len(buf) >= self._s.min_samples:
            hv = np.array([(f.h, f.v) for f in buf])
            mask = mad_inliers(hv)
            inliers = [f for f, ok in zip(buf, mask) if ok]
            if len(inliers) >= self._s.min_samples:
                spread = np.median(np.abs(hv[mask] - np.median(hv[mask], axis=0)), axis=0) * 1.4826
                accepted = bool(np.all(spread <= self._spread_limit()))
            if accepted:
                target = self.targets[self._index]
                self.samples.extend(CalibrationSample.from_features(f, target, self._index) for f in inliers)
                self._ears.extend((f.ear_left, f.ear_right) for f in inliers)
                self._lumas.extend(f.face_luma for f in inliers)
                if not self._is_pose_point():  # a faixa de referência vem do passe 1
                    self._medians.append(np.median(hv[mask], axis=0))

        if not accepted and self._retry < self._s.max_retries:
            self._retry += 1
            self._set_phase(Phase.SETTLE)
            return
        if not accepted:
            self.skipped.append(self._index)
        self._index += 1
        self._retry = 0
        self._pose_wait = 0.0
        if self.pose_start is not None and self._index == self.pose_start:
            self._compute_pose_ref()
        if self._index >= len(self.targets):
            needed = 6 if self._s.polynomial_degree == 2 else 3
            self._phase = Phase.DONE if len(self.targets) - len(self.skipped) >= needed else Phase.FAILED
        else:
            self._set_phase(Phase.SETTLE)
