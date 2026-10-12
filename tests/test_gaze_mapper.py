import numpy as np
import pytest

from lorena.domain.models import CalibrationSample, GazeFeatures
from lorena.mapping.gaze_mapper import GazeMapper, NotFittedError


def test_mapper_not_fitted_raises():
    mapper = GazeMapper(degree=2, screen_size=(1920, 1080))
    with pytest.raises(NotFittedError):
        mapper.predict_raw(0.5, 0.5)


def test_mapper_fit_and_predict_roundtrip():
    mapper = GazeMapper(degree=2, ridge_lambda=1e-5, screen_size=(1920, 1080))

    # Create 9 points on screen
    xs = [100.0, 960.0, 1820.0]
    ys = [100.0, 540.0, 980.0]
    targets = [(x, y) for y in ys for x in xs]

    samples = []
    # Simulate h in [0.3, 0.7], v in [-0.2, 0.2] with quadratic mapping
    for idx, (tx, ty) in enumerate(targets):
        h = 0.3 + 0.4 * (tx - 100.0) / 1720.0
        v = -0.2 + 0.4 * (ty - 100.0) / 880.0
        # Add 10 samples per point with small noise
        for _ in range(10):
            samples.append(CalibrationSample(
                h=h + np.random.normal(0, 0.001),
                v=v + np.random.normal(0, 0.001),
                yaw=0.0,
                pitch=0.0,
                target_x=tx,
                target_y=ty,
                point_index=idx
            ))

    report = mapper.fit(samples)
    assert mapper.is_fitted
    assert report.num_points == 9
    assert report.mean_error_px < 20.0  # High accuracy on clean synthetic data

    # Test prediction
    pred = mapper.predict_raw(0.5, 0.0)
    assert 900.0 <= pred[0] <= 1020.0
    assert 500.0 <= pred[1] <= 580.0

    # Serialization roundtrip
    data = mapper.to_dict()
    restored = GazeMapper.from_dict(data)
    assert restored.is_fitted
    np.testing.assert_allclose(restored.predict_raw(0.5, 0.0), pred, atol=1e-5)
