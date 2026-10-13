import numpy as np
import pytest

from lorena.domain.models import GazeFeatures
from lorena.mapping.gaze_mapper import GazeMapper
from lorena.services.accuracy_evaluator import AccuracyEvaluator


def test_accuracy_evaluator_regional_classification():
    mapper = GazeMapper(degree=1, auto_select=False, screen_size=(1920, 1080))
    mapper._mu = np.array([0.5, 0.0])
    mapper._sigma = np.array([0.1, 0.1])
    mapper._beta = np.array([[960.0, 540.0], [500.0, 0.0], [0.0, 300.0]])
    mapper._zmin = np.array([-2.0, -2.0])
    mapper._zmax = np.array([2.0, 2.0])

    evaluator = AccuracyEvaluator(mapper, screen_size=(1920, 1080), viewing_distance_mm=600.0, pixel_pitch_mm=0.25)

    # Classifications
    assert evaluator.classify_region(100.0, 100.0) == "corner"
    assert evaluator.classify_region(960.0, 50.0) == "edge"
    assert evaluator.classify_region(960.0, 540.0) == "center"

    # Evaluation with targets in different regions
    sample = GazeFeatures(
        timestamp=1.0, h=0.5, v=0.0, yaw=0.0, pitch=0.0,
        ear_left=0.3, ear_right=0.3, openness_left=1.0, openness_right=1.0, quality=1.0
    )
    # Mapper predicts (960, 540)
    evaluations = [
        ((960.0, 540.0), [sample]),       # Center: error = 0 px
        ((100.0, 100.0), [sample]),       # Corner: error = ||(960,540) - (100,100)|| ≈ 966 px
    ]
    report = evaluator.evaluate_points(evaluations)
    assert "center" in report.region_error_px
    assert "corner" in report.region_error_px
    assert report.region_error_px["center"] == pytest.approx(0.0, abs=1e-3)
    assert report.region_error_px["corner"] > 900.0
