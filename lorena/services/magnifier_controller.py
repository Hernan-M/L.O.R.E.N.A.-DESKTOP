"""Controlador e máquina de estados da Lupa Inteligente (Assistive Zoom Lens).

Gerencia a detecção de fixação (>1.5s), progresso de engate pré-lupa (1.0s a 1.5s),
remapeamento de coordenadas de clique ampliado para a tela real e escape ao desviar o olhar.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from lorena.config import MagnifierSettings
from lorena.mapping.filters import FixationStabilizer


class MagnifierState(str, Enum):
    IDLE = "idle"
    CHARGING = "charging"   # Fixação entre 1.0s e 1.5s: indicador de engate
    ACTIVE = "active"       # Lupa aberta na tela


def ease_in_cubic(t: float) -> float:
    """Interpolação cúbica de aceleração suave de 0.0 a 1.0."""
    t_clamped = max(0.0, min(1.0, float(t)))
    return 1.0 - math.pow(1.0 - t_clamped, 3)


def ease_out_cubic(t: float) -> float:
    """Interpolação cúbica de desaceleração suave de 1.0 a 0.0."""
    t_clamped = max(0.0, min(1.0, float(t)))
    return math.pow(1.0 - t_clamped, 3)


@dataclass(frozen=True)
class MagnifierStatus:
    state: MagnifierState
    center: tuple[float, float] | None = None
    charge_progress: float = 0.0       # 0.0 a 1.0 (anel de pré-ativação)
    zoom_factor: float = 2.2
    lens_radius_px: int = 130
    real_target: tuple[float, float] | None = None


class MagnifierController:
    """Máquina de estados para ativação, navegação e clique por lupa de precisão."""

    def __init__(self, settings: MagnifierSettings) -> None:
        self._s = settings
        self.state = MagnifierState.IDLE
        self.center: np.ndarray | None = None
        self._opened_for_fixation: float | None = None
        self._escape_since: float | None = None

    def reset(self) -> None:
        self.state = MagnifierState.IDLE
        self.center = None
        self._opened_for_fixation = None
        self._escape_since = None

    def dismiss(self) -> None:
        """Fecha a lupa imediatamente (ex.: após clique ou sacada)."""
        self.reset()

    def map_to_real_screen(self, gaze_pos: np.ndarray) -> np.ndarray:
        """Converte uma posição olhada dentro da lente na coordenada real ampliada."""
        if self.center is None:
            return gaze_pos
        offset = np.asarray(gaze_pos, dtype=float) - self.center
        real = self.center + offset / self._s.zoom_factor
        return real

    def update(self, stabilizer: FixationStabilizer, gaze_pos: np.ndarray, t: float) -> MagnifierStatus:
        if not self._s.enabled:
            if self.state is not MagnifierState.IDLE:
                self.reset()
            return MagnifierStatus(MagnifierState.IDLE)

        # Se a lupa já estiver ativa
        if self.state is MagnifierState.ACTIVE and self.center is not None:
            dist = float(np.linalg.norm(gaze_pos - self.center))
            escape_radius = self._s.lens_radius_px + self._s.escape_margin_px

            # Checa se o usuário desviou o olhar para longe da lupa
            if dist > escape_radius:
                if self._escape_since is None:
                    self._escape_since = t
                elif t - self._escape_since >= 0.30:  # 300 ms fora da lente fecha
                    self.reset()
                    return MagnifierStatus(MagnifierState.IDLE)
            else:
                self._escape_since = None

            real_target = self.map_to_real_screen(gaze_pos)
            return MagnifierStatus(
                state=MagnifierState.ACTIVE,
                center=(float(self.center[0]), float(self.center[1])),
                charge_progress=1.0,
                zoom_factor=self._s.zoom_factor,
                lens_radius_px=self._s.lens_radius_px,
                real_target=(float(real_target[0]), float(real_target[1])),
            )

        # Lupa inativa: avalia duração de fixação contínua
        duration = stabilizer.fixation_duration(t)
        if stabilizer.center is None or duration <= 0:
            self.reset()
            return MagnifierStatus(MagnifierState.IDLE)

        # Evita reabrir repetidamente para a mesma fixação já processada
        if self._opened_for_fixation == stabilizer.fixation_start:
            return MagnifierStatus(MagnifierState.IDLE)

        # Fase 1: Anel de pré-engate (entre charge_dwell_s e trigger_dwell_s)
        t_charge = self._s.charge_dwell_s
        t_trigger = self._s.trigger_dwell_s
        if duration >= t_charge and duration < t_trigger:
            progress = min(max((duration - t_charge) / max(t_trigger - t_charge, 1e-3), 0.0), 1.0)
            self.state = MagnifierState.CHARGING
            return MagnifierStatus(
                state=MagnifierState.CHARGING,
                center=(float(stabilizer.center[0]), float(stabilizer.center[1])),
                charge_progress=progress,
                zoom_factor=self._s.zoom_factor,
                lens_radius_px=self._s.lens_radius_px,
            )

        # Fase 2: Fixação ultrapassou trigger_dwell_s (>1.5s) -> Dispara a Lupa!
        if duration >= t_trigger:
            self.state = MagnifierState.ACTIVE
            self.center = stabilizer.center.copy()
            self._opened_for_fixation = stabilizer.fixation_start
            self._escape_since = None
            real_target = self.map_to_real_screen(gaze_pos)
            return MagnifierStatus(
                state=MagnifierState.ACTIVE,
                center=(float(self.center[0]), float(self.center[1])),
                charge_progress=1.0,
                zoom_factor=self._s.zoom_factor,
                lens_radius_px=self._s.lens_radius_px,
                real_target=(float(real_target[0]), float(real_target[1])),
            )

        self.state = MagnifierState.IDLE
        return MagnifierStatus(MagnifierState.IDLE)
