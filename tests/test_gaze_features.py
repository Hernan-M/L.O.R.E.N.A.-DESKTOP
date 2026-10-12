import math
import numpy as np
import pytest

from lorena.config import BlinkSettings
from lorena.domain.models import EyeObservation, FaceObservation
from lorena.vision.gaze_features import GazeFeatureExtractor, OpennessModel, eye_ratios


def test_eye_ratios_centered():
    start = np.array([100.0, 100.0])
    end = np.array([140.0, 100.0])
    mid = (start + end) / 2.0  # [120, 100]

    h, v = eye_ratios(mid, start, end)
    assert pytest.approx(h, abs=1e-5) == 0.5
    assert pytest.approx(v, abs=1e-5) == 0.0


def test_eye_ratios_scale_and_rotation_invariance():
    start = np.array([50.0, 50.0])
    end = np.array([100.0, 50.0])
    iris = np.array([75.0, 55.0])  # slightly shifted down

    h1, v1 = eye_ratios(iris, start, end)

    # Scale 2x
    h2, v2 = eye_ratios(iris * 2.0, start * 2.0, end * 2.0)
    assert pytest.approx(h1, abs=1e-5) == h2
    assert pytest.approx(v1, abs=1e-5) == v2

    # Rotate 45 degrees
    theta = math.radians(45.0)
    rot = np.array([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta),  math.cos(theta)]
    ])
    h3, v3 = eye_ratios(rot @ iris, rot @ start, rot @ end)
    assert pytest.approx(h1, abs=1e-5) == h3
    assert pytest.approx(v1, abs=1e-5) == v3


def test_openness_model():
    settings = BlinkSettings(default_open_ear=0.30, closed_ratio=0.35)
    model = OpennessModel(settings, open_left=0.30, open_right=0.30)

    # Open EAR gives 1.0
    assert pytest.approx(model.openness(0.30, 0.30), abs=1e-3) == 1.0

    # Closed EAR (0.30 * 0.35 = 0.105) gives 0.0
    assert pytest.approx(model.openness(0.105, 0.30), abs=1e-3) == 0.0

    # Intermediate gives ~0.5
    assert pytest.approx(model.openness(0.2025, 0.30), abs=1e-2) == 0.5


def test_gaze_feature_extractor_weighted():
    settings = BlinkSettings(default_open_ear=0.30, closed_ratio=0.35)
    model = OpennessModel(settings, open_left=0.30, open_right=0.30)
    extractor = GazeFeatureExtractor(model)

    start_l = np.array([100.0, 100.0])
    end_l = np.array([140.0, 100.0])
    iris_l = np.array([120.0, 100.0])  # h=0.5, v=0.0

    start_r = np.array([200.0, 100.0])
    end_r = np.array([240.0, 100.0])
    iris_r = np.array([228.0, 100.0])  # h=0.7, v=0.0

    # Left eye open, right eye closed
    left_eye = EyeObservation(iris=iris_l, start_corner=start_l, end_corner=end_l, ear=0.30)
    right_eye = EyeObservation(iris=iris_r, start_corner=start_r, end_corner=end_r, ear=0.10)
    face = FaceObservation(timestamp=1.0, left_eye=left_eye, right_eye=right_eye)

    features = extractor.extract(face)
    assert features is not None
    # Since right eye is closed (openness ~0), left eye dominates (h ~ 0.5)
    assert pytest.approx(features.h, abs=0.05) == 0.5
    assert features.quality > 0.4
