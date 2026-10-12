"""Objetos de domínio imutáveis trocados entre as camadas.

Convenção de lateralidade: ``left``/``right`` sempre se referem ao olho
ANATÔMICO do usuário (não ao lado da imagem). A detecção roda sobre o frame
não espelhado, portanto os índices do MediaPipe já são anatômicos.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


@dataclass(frozen=True, slots=True)
class EyeObservation:
    """Geometria de um olho em pixels (float, subpixel).

    ``start_corner``/``end_corner`` são os cantos ordenados da esquerda para a
    direita **da imagem**, de forma que o eixo start→end aponta sempre para a
    direita da imagem nos dois olhos (razões horizontais comparáveis).
    """

    iris: np.ndarray
    start_corner: np.ndarray
    end_corner: np.ndarray
    ear: float


@dataclass(frozen=True, slots=True)
class FaceObservation:
    timestamp: float
    left_eye: EyeObservation
    right_eye: EyeObservation
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0


@dataclass(frozen=True, slots=True)
class GazeFeatures:
    """Características normalizadas do olhar usadas pelo mapeador."""

    timestamp: float
    h: float
    v: float
    yaw: float
    pitch: float
    ear_left: float
    ear_right: float
    openness_left: float
    openness_right: float
    quality: float


@dataclass(frozen=True, slots=True)
class CalibrationSample:
    h: float
    v: float
    yaw: float
    pitch: float
    target_x: float
    target_y: float
    point_index: int

    @classmethod
    def from_features(cls, f: GazeFeatures, target: tuple[float, float], index: int) -> "CalibrationSample":
        return cls(f.h, f.v, f.yaw, f.pitch, float(target[0]), float(target[1]), index)


class TrackingState(str, Enum):
    IDLE = "idle"
    TRACKING = "tracking"
    DEGRADED = "degraded"
    LOST = "lost"
    PAUSED = "paused"


class GestureEvent(str, Enum):
    CLICK = "click"
    RIGHT_CLICK = "right_click"
    TOGGLE_PAUSE = "toggle_pause"
    DWELL_CLICK = "dwell_click"


@dataclass(frozen=True, slots=True)
class SessionStatus:
    state: TrackingState
    dwell_progress: float = 0.0
    last_event: GestureEvent | None = None


@dataclass(frozen=True, slots=True)
class FrameResult:
    """O que o pipeline publica para a UI a cada frame processado."""

    timestamp: float
    features: GazeFeatures | None
    fps: float
    status: SessionStatus | None = None
    stopped_by_user: bool = False
    error: str | None = None
    extra: dict = field(default_factory=dict)
