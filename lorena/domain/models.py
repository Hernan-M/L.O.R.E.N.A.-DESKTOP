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
    upper_lid: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class FaceObservation:
    timestamp: float
    left_eye: EyeObservation
    right_eye: EyeObservation
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    head_x: float = 0.0   # tx/tz da matriz facial (adimensional)
    head_y: float = 0.0   # ty/tz
    head_z: float = 0.0   # tz (cm, unidade da matriz do MediaPipe)
    luma: float = 0.0     # luminancia media (0-255) do ROI do rosto


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
    lid: float = 0.0       # pos. da palpebra superior / largura do olho
    eye_px: float = 0.0    # largura media do olho em pixels
    head_x: float = 0.0
    head_y: float = 0.0
    head_z: float = 0.0
    face_luma: float = 0.0


@dataclass(frozen=True, slots=True)
class CalibrationSample:
    h: float
    v: float
    yaw: float
    pitch: float
    target_x: float
    target_y: float
    point_index: int
    lid: float = 0.0
    head_x: float = 0.0
    head_y: float = 0.0

    @classmethod
    def from_features(cls, f: GazeFeatures, target: tuple[float, float], index: int) -> "CalibrationSample":
        return cls(f.h, f.v, f.yaw, f.pitch, float(target[0]), float(target[1]), index, f.lid, f.head_x, f.head_y)


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
    hint: str | None = None   # "light" | "head" | None
    magnifier: Any = None     # MagnifierStatus quando a lupa estiver ativa/carregando


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
