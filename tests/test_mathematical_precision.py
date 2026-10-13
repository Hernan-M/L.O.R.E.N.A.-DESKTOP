import numpy as np
import pytest

from lorena.domain.models import CalibrationSample
from lorena.mapping.gaze_mapper import GazeMapper
from lorena.vision import landmarks as lm
from lorena.vision.face_detector import face_from_landmarks


def test_iris_5point_centroid_averaging():
    points = np.zeros((478, 2))
    # Right iris center at (100, 50)
    points[lm.IRIS_CENTERS[0]] = [100.0, 50.0]
    # Ring points around it
    points[lm.IRIS_RING_A[0]] = [104.0, 50.0]
    points[lm.IRIS_RING_A[1]] = [100.0, 54.0]
    points[lm.IRIS_RING_A[2]] = [96.0, 50.0]
    points[lm.IRIS_RING_A[3]] = [100.0, 46.0]

    # Left eye landmarks
    points[lm.LEFT_EYE_CORNERS[0]] = [200.0, 50.0]
    points[lm.LEFT_EYE_CORNERS[1]] = [240.0, 50.0]
    points[lm.IRIS_CENTERS[1]] = [220.0, 50.0]
    for idx in lm.IRIS_RING_B:
        points[idx] = [220.0, 50.0]

    # Right eye corners
    points[lm.RIGHT_EYE_CORNERS[0]] = [80.0, 50.0]
    points[lm.RIGHT_EYE_CORNERS[1]] = [120.0, 50.0]

    obs = face_from_landmarks(points, 1.0)
    assert obs is not None
    # Centroid of the right iris must be exactly (100.0, 50.0)
    assert np.allclose(obs.right_eye.iris, [100.0, 50.0])
    assert np.allclose(obs.left_eye.iris, [220.0, 50.0])


def test_decoupled_biomechanical_regression_orthogonality():
    # 14 points (9 at rest + 5 with head pose)
    xs = [50.0, 683.0, 1316.0]
    ys = [50.0, 384.0, 718.0]
    targets = [(x, y) for y in ys for x in xs]
    targets += [(683.0, 384.0), (50.0, 50.0), (1316.0, 50.0), (50.0, 718.0), (1316.0, 718.0)]

    samples = []
    for idx, (tx, ty) in enumerate(targets):
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

    mapper = GazeMapper(degree=1, use_head_pose=True, auto_select=True, screen_size=(1366, 768))
    report = mapper.fit(samples)
    assert mapper.is_fitted
    assert report.loo_mean_error_px is not None
    assert report.loo_mean_error_px < 5.0  # Decoupled LOO error is well below 5 px

    # Head coefficients: yaw -> X and pitch -> Y are decoupled (zero cross-talk)
    # Rows: 0 (intercept), 1 (h), 2 (v), 3 (yaw), 4 (pitch), 5 (hx), 6 (hy)
    beta = mapper._beta
    assert beta is not None
    # Pitch effect on X must be exactly zero
    assert beta[4, 0] == pytest.approx(0.0)
    # Yaw effect on Y must be exactly zero
    assert beta[3, 1] == pytest.approx(0.0)
    # hy effect on X must be exactly zero
    assert beta[6, 0] == pytest.approx(0.0)


def test_parity_symmetric_basis_monotonic_corners():
    mapper = GazeMapper(degree=2, basis="symmetric", use_head_pose=False, screen_size=(1920, 1080))
    # Synthetic symmetric points
    targets = [
        (100.0, 100.0), (960.0, 100.0), (1820.0, 100.0),
        (100.0, 540.0), (960.0, 540.0), (1820.0, 540.0),
        (100.0, 980.0), (960.0, 980.0), (1820.0, 980.0),
    ]
    samples = []
    for idx, (tx, ty) in enumerate(targets):
        # Cubic/odd expansion
        u_h = (tx - 960.0) / 960.0
        u_v = (ty - 540.0) / 540.0
        h = 0.5 + 0.3 * (u_h + 0.05 * u_h * abs(u_h))
        v = 0.0 + 0.3 * (u_v + 0.05 * u_v * abs(u_v))
        for _ in range(5):
            samples.append(
                CalibrationSample(
                    h=h, v=v, yaw=0.0, pitch=0.0,
                    target_x=tx, target_y=ty, point_index=idx
                )
            )

    report = mapper.fit(samples)
    assert mapper.is_fitted
    assert report.mean_error_px < 15.0
    # Left and right corner errors must be closely balanced
    err_tl = report.per_point_error_px[0]
    err_tr = report.per_point_error_px[2]
    assert abs(err_tl - err_tr) < 10.0
