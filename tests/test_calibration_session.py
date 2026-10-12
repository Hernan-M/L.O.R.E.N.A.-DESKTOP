import numpy as np
import pytest

from lorena.config import CalibrationSettings
from lorena.domain.models import GazeFeatures
from lorena.services.calibration import CalibrationSession, Phase, mad_inliers


def test_mad_inliers():
    # 9 inliers near 1.0, 1 outlier at 10.0
    vals = np.array([
        [1.0, 1.0], [1.02, 0.99], [0.98, 1.01],
        [1.01, 1.0], [0.99, 1.02], [1.03, 0.98],
        [0.97, 1.01], [1.0, 1.0], [1.01, 0.99],
        [10.0, 10.0]
    ])
    mask = mad_inliers(vals)
    assert mask.sum() == 9
    assert not mask[-1]


def test_calibration_session_state_machine():
    targets = [(100, 100), (300, 300), (500, 500)]
    settings = CalibrationSettings(
        settle_s=0.2,
        collect_s=0.3,
        min_samples=3,
        polynomial_degree=1  # 2 points enough for degree 1 in test
    )
    session = CalibrationSession(targets, settings)

    assert session.phase is Phase.WAIT_FACE
    assert not session.finished

    t = 0.0
    dt = 0.05
    # Feeding valid face features
    feat = GazeFeatures(
        timestamp=t, h=0.5, v=0.0, yaw=0.0, pitch=0.0,
        ear_left=0.3, ear_right=0.3, openness_left=1.0, openness_right=1.0, quality=0.9
    )

    # First frame moves to SETTLE
    session.update(feat, t)
    assert session.phase is Phase.SETTLE

    # Progress through SETTLE (0.2s)
    while session.phase is Phase.SETTLE:
        t += dt
        session.update(feat, t)

    assert session.phase is Phase.COLLECT

    # Progress through COLLECT (0.3s)
    while session.phase is Phase.COLLECT:
        t += dt
        session.update(feat, t)

    # Point 0 completed, advances to next point (SETTLE)
    assert session.phase is Phase.SETTLE

    # Complete point 1
    while not session.finished:
        t += dt
        session.update(feat, t)

    assert session.phase is Phase.DONE
    assert len(session.samples) >= 6
