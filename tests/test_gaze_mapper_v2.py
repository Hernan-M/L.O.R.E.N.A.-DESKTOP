import numpy as np
import pytest

from lorena.domain.models import CalibrationSample, GazeFeatures
from lorena.mapping.gaze_mapper import GazeMapper, NotFittedError


def test_mapper_v2_not_fitted():
    mapper = GazeMapper(degree=2, screen_size=(1920, 1080))
    with pytest.raises(NotFittedError):
        mapper.predict_raw(0.5, 0.0)


def test_mapper_v2_auto_select_avoids_overfitting_without_pose_variation():
    # 9 points on screen with no head movement (std(yaw) < MIN_POSE_STD)
    xs = [100.0, 960.0, 1820.0]
    ys = [100.0, 540.0, 980.0]
    targets = [(x, y) for y in ys for x in xs]

    samples = []
    for idx, (tx, ty) in enumerate(targets):
        h = 0.3 + 0.4 * (tx - 100.0) / 1720.0
        v = -0.2 + 0.4 * (ty - 100.0) / 880.0
        for _ in range(10):
            samples.append(
                CalibrationSample(
                    h=h + np.random.normal(0, 0.001),
                    v=v + np.random.normal(0, 0.001),
                    yaw=0.002 * np.random.normal(),  # tiny noise
                    pitch=0.002 * np.random.normal(),
                    target_x=tx,
                    target_y=ty,
                    point_index=idx,
                    lid=0.05,
                    head_x=0.0,
                    head_y=0.0,
                )
            )

    mapper = GazeMapper(degree=2, use_head_pose=True, auto_select=True, screen_size=(1920, 1080))
    report = mapper.fit(samples)
    assert mapper.is_fitted
    # Since head pose had no real variation, model should discard head pose terms
    assert mapper.use_head_pose is False
    assert report.loo_mean_error_px is not None
    assert report.loo_mean_error_px < 40.0  # Well-conditioned LOO error


def test_mapper_v2_bounded_extrapolation():
    # Train simple mapper
    mapper = GazeMapper(degree=1, auto_select=False, screen_size=(1920, 1080))
    samples = []
    targets = [(200, 200), (960, 540), (1720, 880), (200, 880), (1720, 200)]
    for idx, (tx, ty) in enumerate(targets):
        h = 0.4 + 0.2 * (tx / 1920.0)
        v = -0.1 + 0.2 * (ty / 1080.0)
        for _ in range(5):
            samples.append(CalibrationSample(h=h, v=v, yaw=0.0, pitch=0.0, target_x=tx, target_y=ty, point_index=idx))

    mapper.fit(samples)
    # Give extreme outlier far beyond calibrated range
    pred_extreme = mapper.predict_raw(h=5.0, v=-5.0)
    assert 0 <= pred_extreme[0] <= 1920
    assert 0 <= pred_extreme[1] <= 1080


def test_mapper_v2_drift_multi():
    mapper = GazeMapper(degree=1, screen_size=(1920, 1080))
    mapper._mu = np.array([0.5, 0.0])
    mapper._sigma = np.array([0.1, 0.1])
    mapper._beta = np.array([[960.0, 540.0], [500.0, 0.0], [0.0, 300.0]])
    mapper._zmin = np.array([-2.0, -2.0])
    mapper._zmax = np.array([2.0, 2.0])

    f1 = GazeFeatures(timestamp=1.0, h=0.4, v=0.0, yaw=0.0, pitch=0.0, ear_left=0.3, ear_right=0.3, openness_left=1.0, openness_right=1.0, quality=1.0)
    f2 = GazeFeatures(timestamp=1.0, h=0.6, v=0.0, yaw=0.0, pitch=0.0, ear_left=0.3, ear_right=0.3, openness_left=1.0, openness_right=1.0, quality=1.0)

    # Pairs with systematic scale shift
    pairs = [(f1, (400.0, 540.0)), (f2, (1520.0, 540.0))]
    scale, offset = mapper.recalibrate_drift_multi(pairs)
    assert len(scale) == 2
    assert len(offset) == 2
