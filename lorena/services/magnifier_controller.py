"""Controlador e máquina de estados da Lupa Inteligente (Assistive Zoom Lens 2.0).

Gerencia a detecção de fixação com pré-engate ágil (0.7s a 1.0s), ciclo de mira desacoplado
com dwell interno dedicado (0.7s), remapeamento geométrico com compensação de cantos,
live refresh viewport a 10 FPS e escape instantâneo por sacadas rápidas.
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
    CHARGING = "charging"   # Fixação em pré-engate: anel de carregamento
    ACTIVE = "active"       # Lupa aberta na tela: fase de mira e seleção


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
    charge_progress: float = 0.0       # 0.0 a 1.0 (anel de pré-ativação da lente)
    aim_progress: float = 0.0          # 0.0 a 1.0 (anel verde de confirmação de clique na lente)
    zoom_factor: float = 2.2
    lens_radius_px: int = 160
    real_target: tuple[float, float] | None = None
    window_pos: tuple[int, int] | None = None  # Posição (win_x, win_y) no display
    fired_click: bool = False          # True quando o aim dwell atinge 1.0


class MagnifierController:
    """Máquina de estados para ativação, mira desacoplada e clique por lupa de precisão."""

    def __init__(self, settings: MagnifierSettings, screen_size: tuple[int, int] = (1920, 1080)) -> None:
        self._s = settings
        self._screen_size = screen_size
        self.state = MagnifierState.IDLE
        self.center: np.ndarray | None = None
        self.window_pos: tuple[int, int] | None = None
        self.roi_center: np.ndarray | None = None
        self._opened_for_fixation: float | None = None
        self._escape_since: float | None = None
        self._aim_start_t: float | None = None
        self._click_fired: bool = False
        self._last_gaze: np.ndarray | None = None
        self._last_gaze_t: float | None = None
        self._gaze_speed: float = 0.0

    def reset(self) -> None:
        self.state = MagnifierState.IDLE
        self.center = None
        self.window_pos = None
        self.roi_center = None
        self._opened_for_fixation = None
        self._escape_since = None
        self._aim_start_t = None
        self._click_fired = False
        self._last_gaze = None
        self._last_gaze_t = None
        self._gaze_speed = 0.0

    def dismiss(self) -> None:
        """Fecha a lupa imediatamente (ex.: após clique ou sacada)."""
        self.reset()

    def set_screen_size(self, size: tuple[int, int]) -> None:
        self._screen_size = size

    def map_to_real_screen(self, gaze_pos: np.ndarray) -> np.ndarray:
        """Converte uma posição olhada dentro da lente na coordenada real ampliada."""
        if self.center is None:
            return gaze_pos

        # Se houver geometria clampeada precisa com roi_center e window_pos:
        if self.roi_center is not None and self.window_pos is not None:
            D = float(self._s.lens_radius_px * 2)
            wx, wy = self.window_pos
            lx = float(gaze_pos[0]) - float(wx)
            ly = float(gaze_pos[1]) - float(wy)
            real_x = self.roi_center[0] + (lx - D / 2.0) / self._s.zoom_factor
            real_y = self.roi_center[1] + (ly - D / 2.0) / self._s.zoom_factor
            sw, sh = self._screen_size
            return np.array([
                np.clip(real_x, 0.0, float(sw - 1)),
                np.clip(real_y, 0.0, float(sh - 1)),
            ], dtype=float)

        # Fallback analítico direto centrado na âncora:
        offset = np.asarray(gaze_pos, dtype=float) - self.center
        real = self.center + offset / self._s.zoom_factor
        return real

    def update(self, stabilizer: FixationStabilizer, gaze_pos: np.ndarray, t: float) -> MagnifierStatus:
        if not self._s.enabled:
            if self.state is not MagnifierState.IDLE:
                self.reset()
            return MagnifierStatus(MagnifierState.IDLE)

        # Cálculo da velocidade instantânea do olhar (para escape por sacada)
        if self._last_gaze is not None and self._last_gaze_t is not None:
            dt = max(t - self._last_gaze_t, 1e-3)
            self._gaze_speed = float(np.linalg.norm(gaze_pos - self._last_gaze)) / dt
        self._last_gaze = np.asarray(gaze_pos, dtype=float).copy()
        self._last_gaze_t = t

        # Se a lupa já estiver ativa
        if self.state is MagnifierState.ACTIVE and self.center is not None:
            # 1. Escape por Distância fora da Lente:
            if self.window_pos is not None:
                wx, wy = self.window_pos
                r = float(self._s.lens_radius_px)
                lens_center = np.array([wx + r, wy + r], dtype=float)
            else:
                lens_center = self.center

            dist = float(np.linalg.norm(gaze_pos - lens_center))
            escape_radius = self._s.lens_radius_px + self._s.escape_margin_px

            # Sacada ampla para longe (> 2.5x escape radius): fecha imediatamente
            if dist > escape_radius * 2.5:
                self.reset()
                return MagnifierStatus(MagnifierState.IDLE)

            # Desvio para fora da lente (> escape radius): fecha após 250 ms de permanência fora
            if dist > escape_radius:
                if self._escape_since is None:
                    self._escape_since = t
                elif t - self._escape_since >= 0.25:  # 250 ms fora da lente fecha
                    self.reset()
                    return MagnifierStatus(MagnifierState.IDLE)
            else:
                self._escape_since = None

            # 3. Ciclo de Mira Desacoplado (Aim Dwell):
            if self._aim_start_t is None:
                self._aim_start_t = t

            aim_duration = max(0.0, t - self._aim_start_t)
            aim_dwell_s = getattr(self._s, "aim_dwell_s", 0.7)
            aim_progress = min(max(aim_duration / max(aim_dwell_s, 1e-3), 0.0), 1.0)
            fired_click = False

            if aim_progress >= 1.0 and not self._click_fired:
                fired_click = True
                self._click_fired = True

            real_target = self.map_to_real_screen(gaze_pos)
            return MagnifierStatus(
                state=MagnifierState.ACTIVE,
                center=(float(self.center[0]), float(self.center[1])),
                charge_progress=1.0,
                aim_progress=aim_progress,
                zoom_factor=self._s.zoom_factor,
                lens_radius_px=self._s.lens_radius_px,
                real_target=(float(real_target[0]), float(real_target[1])),
                window_pos=self.window_pos,
                fired_click=fired_click,
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

        # Fase 2: Fixação ultrapassou trigger_dwell_s -> Dispara a Lupa 2.0!
        if duration >= t_trigger:
            sw, sh = self._screen_size
            D = int(self._s.lens_radius_px * 2)
            cx = float(stabilizer.center[0])
            cy = float(stabilizer.center[1])

            # Clampeamento da janela no display
            win_x = max(0, min(sw - D, int(cx - D // 2)))
            win_y = max(0, min(sh - D, int(cy - D // 2)))
            self.window_pos = (win_x, win_y)

            # Cálculo da ROI na tela real
            zoom = max(1.0, float(self._s.zoom_factor))
            roi_size = max(32, int(round(D / zoom)))
            src_x = max(0, min(sw - roi_size, int(cx - roi_size // 2)))
            src_y = max(0, min(sh - roi_size, int(cy - roi_size // 2)))
            self.roi_center = np.array([src_x + roi_size / 2.0, src_y + roi_size / 2.0], dtype=float)

            self.state = MagnifierState.ACTIVE
            self.center = stabilizer.center.copy()
            self._opened_for_fixation = stabilizer.fixation_start
            self._escape_since = None
            self._aim_start_t = t
            self._click_fired = False

            real_target = self.map_to_real_screen(gaze_pos)
            return MagnifierStatus(
                state=MagnifierState.ACTIVE,
                center=(float(self.center[0]), float(self.center[1])),
                charge_progress=1.0,
                aim_progress=0.0,
                zoom_factor=self._s.zoom_factor,
                lens_radius_px=self._s.lens_radius_px,
                real_target=(float(real_target[0]), float(real_target[1])),
                window_pos=self.window_pos,
                fired_click=False,
            )

        self.state = MagnifierState.IDLE
        return MagnifierStatus(MagnifierState.IDLE)
