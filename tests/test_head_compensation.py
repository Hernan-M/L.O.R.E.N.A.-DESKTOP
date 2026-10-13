import math
import numpy as np
import pytest

from lorena.config import Settings
from lorena.domain.models import CalibrationSample, GazeFeatures, TrackingState
from lorena.mapping.gaze_mapper import GazeMapper
from lorena.services.cursor import Cursor
from lorena.services.cursor_session import CursorSession


class MockCursor(Cursor):
    def __init__(self):
        self.pos = np.array([0.0, 0.0])
        self.clicks = []

    def move_to(self, pos: np.ndarray) -> None:
        self.pos = np.array(pos, dtype=float)

    def click(self, button: str = "left") -> None:
        self.clicks.append(button)


def test_head_rate_gate_freezes_cursor_on_rapid_turn():
    mapper = GazeMapper(degree=1, auto_select=False, screen_size=(1920, 1080))
    mapper._mu = np.array([0.5, 0.0])
    mapper._sigma = np.array([0.1, 0.1])
    mapper._beta = np.array([[960.0, 540.0], [500.0, 0.0], [0.0, 300.0]])
    mapper._zmin = np.array([-2.0, -2.0])
    mapper._zmax = np.array([2.0, 2.0])

    cursor = MockCursor()
    settings = Settings()
    session = CursorSession(mapper, cursor, settings)

    # Frame 1: Stable head pose
    f1 = GazeFeatures(
        timestamp=1.0, h=0.5, v=0.0, yaw=0.0, pitch=0.0,
        ear_left=0.3, ear_right=0.3, openness_left=1.0, openness_right=1.0, quality=1.0
    )
    status1 = session.on_frame(f1, 1.0)
    pos1 = cursor.pos.copy()
    assert np.allclose(pos1, [960.0, 540.0])

    # Frame 2: Rapid head turn (10 degrees in 0.05s = 200 deg/s > 60 deg/s threshold)
    yaw_jump = math.radians(10.0)
    f2 = GazeFeatures(
        timestamp=1.05, h=0.6, v=0.0, yaw=yaw_jump, pitch=0.0,
        ear_left=0.3, ear_right=0.3, openness_left=1.0, openness_right=1.0, quality=1.0
    )
    status2 = session.on_frame(f2, 1.05)
    # Cursor should NOT have moved during the rapid head turn (hold active)
    assert np.allclose(cursor.pos, pos1)


def test_head_pose_and_translation_model_fit():
    # 14 points (9 pass 1 + 5 pass 2) with simulated 15.6" screen (1366x768)
    mapper = GazeMapper(degree=1, use_head_pose=True, auto_select=True, screen_size=(1366, 768))

    xs = [50.0, 683.0, 1316.0]
    ys = [50.0, 384.0, 718.0]
    targets = [(x, y) for y in ys for x in xs]
    # Pass 2 points (center + 4 corners)
    targets += [(683.0, 384.0), (50.0, 50.0), (1316.0, 50.0), (50.0, 718.0), (1316.0, 718.0)]

    samples = []
    for idx, (tx, ty) in enumerate(targets):
        # Variando a pose no passe 2 (idx >= 9): o olho na cabeça (h, v) compensa o giro da cabeça
        yaw = float(np.sin(idx)) * 0.08 if idx >= 9 else 0.001
        pitch = float(np.cos(idx)) * 0.08 if idx >= 9 else 0.001
        hx = 0.05 if idx >= 9 else 0.0
        h = 0.35 + 0.3 * (tx / 1366.0) - 0.25 * yaw
        v = -0.15 + 0.3 * (ty / 768.0) - 0.25 * pitch
        for _ in range(8):
            samples.append(
                CalibrationSample(
                    h=h + np.random.normal(0, 0.001),
                    v=v + np.random.normal(0, 0.001),
                    yaw=yaw,
                    pitch=pitch,
                    target_x=tx,
                    target_y=ty,
                    point_index=idx,
                    lid=0.04,
                    head_x=hx,
                    head_y=0.0,
                )
            )

    report = mapper.fit(samples)
    assert mapper.is_fitted
    assert report.num_points == 14
    # With pose variation in pass 2, head pose should be included
    assert mapper.use_head_pose is True
    assert report.mean_error_px < 35.0


def test_kinematic_prediction_saccade_and_fixation():
    mapper = GazeMapper(degree=1, auto_select=False, screen_size=(1920, 1080))
    mapper._mu = np.array([0.5, 0.0])
    mapper._sigma = np.array([0.1, 0.1])
    mapper._beta = np.array([[960.0, 540.0], [1000.0, 0.0], [0.0, 500.0]])
    mapper._zmin = np.array([-5.0, -5.0])
    mapper._zmax = np.array([5.0, 5.0])

    cursor = MockCursor()
    settings = Settings()
    settings.filters.kinematic_prediction = True
    settings.filters.lead_time_s = 0.020
    settings.filters.saccade_velocity_threshold = 80.0
    session = CursorSession(mapper, cursor, settings)

    # 1. Repouso (fixação): velocidade zero -> sem projeção de avanço (w = 0)
    f1 = GazeFeatures(
        timestamp=1.00, h=0.5, v=0.0, yaw=0.0, pitch=0.0,
        ear_left=0.3, ear_right=0.3, openness_left=1.0, openness_right=1.0, quality=1.0
    )
    session.on_frame(f1, 1.00)
    pos1 = cursor.pos.copy()
    assert np.allclose(pos1, [960.0, 540.0])

    f2 = GazeFeatures(
        timestamp=1.033, h=0.5, v=0.0, yaw=0.0, pitch=0.0,
        ear_left=0.3, ear_right=0.3, openness_left=1.0, openness_right=1.0, quality=1.0
    )
    session.on_frame(f2, 1.033)
    pos2 = cursor.pos.copy()
    assert np.allclose(pos2, [960.0, 540.0])

    # 2. Sacada rápida para a direita: h cresce suavemente gerando velocidade > 80 px/s
    f3 = GazeFeatures(
        timestamp=1.066, h=0.51, v=0.0, yaw=0.0, pitch=0.0,
        ear_left=0.3, ear_right=0.3, openness_left=1.0, openness_right=1.0, quality=1.0
    )
    session.on_frame(f3, 1.066)

    f4 = GazeFeatures(
        timestamp=1.099, h=0.52, v=0.0, yaw=0.0, pitch=0.0,
        ear_left=0.3, ear_right=0.3, openness_left=1.0, openness_right=1.0, quality=1.0
    )
    session.on_frame(f4, 1.099)

    # Cursor despachado deve estar adiantado à frente da posição estável do histórico
    unprojected = session._history.at(1.099)
    assert unprojected is not None
    assert cursor.pos[0] > unprojected[0]


def test_kinematic_prediction_disabled():
    mapper = GazeMapper(degree=1, auto_select=False, screen_size=(1920, 1080))
    mapper._mu = np.array([0.5, 0.0])
    mapper._sigma = np.array([0.1, 0.1])
    mapper._beta = np.array([[960.0, 540.0], [1000.0, 0.0], [0.0, 500.0]])
    mapper._zmin = np.array([-5.0, -5.0])
    mapper._zmax = np.array([5.0, 5.0])

    cursor = MockCursor()
    settings = Settings()
    settings.filters.kinematic_prediction = False
    session = CursorSession(mapper, cursor, settings)

    f1 = GazeFeatures(
        timestamp=1.00, h=0.5, v=0.0, yaw=0.0, pitch=0.0,
        ear_left=0.3, ear_right=0.3, openness_left=1.0, openness_right=1.0, quality=1.0
    )
    session.on_frame(f1, 1.00)

    f2 = GazeFeatures(
        timestamp=1.033, h=0.60, v=0.0, yaw=0.0, pitch=0.0,
        ear_left=0.3, ear_right=0.3, openness_left=1.0, openness_right=1.0, quality=1.0
    )
    session.on_frame(f2, 1.033)

    unprojected = session._history.at(1.033)
    assert unprojected is not None
    # Com predição desativada, cursor despachado é estritamente igual à posição filtrada
    assert np.allclose(cursor.pos, unprojected)

