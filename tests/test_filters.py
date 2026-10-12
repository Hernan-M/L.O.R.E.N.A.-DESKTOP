import numpy as np
import pytest

from lorena.mapping.filters import (
    DwellClicker,
    FixationStabilizer,
    OneEuroFilter,
    OneEuroFilter2D,
    OutlierGate,
    PositionHistory,
)


def test_one_euro_filter_reduces_jitter():
    # Low cutoff, small beta -> strong smoothing on stationary signal
    f = OneEuroFilter(min_cutoff=0.5, beta=0.001)

    t = 0.0
    dt = 0.033  # ~30 FPS
    values = [500.0 + (5.0 if i % 2 == 0 else -5.0) for i in range(30)]

    filtered = []
    for v in values:
        filtered.append(f(v, t))
        t += dt

    # Filtered variance should be much smaller than raw variance
    assert np.std(filtered[10:]) < np.std(values[10:]) * 0.4


def test_outlier_gate_rejects_single_spike():
    gate = OutlierGate(jump_px=100.0, confirm_frames=3)

    p0 = np.array([500.0, 500.0])
    assert np.allclose(gate(p0), p0)

    # Spike of 300px
    p_spike = np.array([800.0, 500.0])
    res_spike = gate(p_spike)
    # Should maintain previous stable reference
    assert np.allclose(res_spike, p0)

    # Returns to normal
    assert np.allclose(gate(p0), p0)


def test_outlier_gate_accepts_sustained_movement():
    gate = OutlierGate(jump_px=100.0, confirm_frames=3)

    p0 = np.array([500.0, 500.0])
    gate(p0)

    p_new = np.array([800.0, 500.0])
    # 1st frame of jump -> holds p0
    assert np.allclose(gate(p_new), p0)
    # 2nd frame of jump -> holds p0
    assert np.allclose(gate(p_new), p0)
    # 3rd frame of jump -> confirms jump to p_new
    res = gate(p_new)
    assert np.allclose(res, p_new)


def test_fixation_stabilizer_and_dwell():
    stabilizer = FixationStabilizer(radius_px=30.0)
    dwell = DwellClicker(dwell_s=0.5)

    p_center = np.array([500.0, 500.0])

    t = 0.0
    # First frame initializes
    res = stabilizer(p_center, t)
    assert np.allclose(res, p_center)

    # Micro-movements within 30px deadzone
    t = 0.2
    p_jitter = np.array([515.0, 505.0])
    res = stabilizer(p_jitter, t)
    assert np.allclose(res, p_center)  # Cursor kept anchored

    fired, progress = dwell.update(stabilizer, t)
    assert not fired
    assert 0.35 <= progress <= 0.45

    # Reach 0.6s of fixation -> dwell should trigger
    t = 0.6
    fired, progress = dwell.update(stabilizer, t)
    assert fired
    assert progress == 1.0


def test_position_history():
    hist = PositionHistory(maxlen=10)

    hist.push(1.0, np.array([100.0, 100.0]))
    hist.push(1.1, np.array([110.0, 110.0]))
    hist.push(1.2, np.array([120.0, 120.0]))

    # Retrieve position at t=1.05 -> should find point at t=1.0
    p = hist.at(1.05)
    assert p is not None
    assert np.allclose(p, np.array([100.0, 100.0]))
