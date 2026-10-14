import math
import numpy as np
import pytest

from lorena.config import MagnifierSettings
from lorena.mapping.filters import FixationStabilizer
from lorena.services.magnifier_controller import (
    MagnifierController,
    MagnifierState,
    ease_in_cubic,
    ease_out_cubic,
)


def test_ease_functions_bounds_and_monotonicity():
    # Bounds: t=0 -> 0 for ease_in, 1 for ease_out; t=1 -> 1 for ease_in, 0 for ease_out
    assert ease_in_cubic(0.0) == pytest.approx(0.0)
    assert ease_in_cubic(1.0) == pytest.approx(1.0)
    assert ease_out_cubic(0.0) == pytest.approx(1.0)
    assert ease_out_cubic(1.0) == pytest.approx(0.0)

    # Clamping for out-of-range t
    assert ease_in_cubic(-0.5) == pytest.approx(0.0)
    assert ease_in_cubic(1.5) == pytest.approx(1.0)
    assert ease_out_cubic(-0.5) == pytest.approx(1.0)
    assert ease_out_cubic(1.5) == pytest.approx(0.0)

    # Monotonicity test
    steps = [i / 20.0 for i in range(21)]
    vals_in = [ease_in_cubic(t) for t in steps]
    vals_out = [ease_out_cubic(t) for t in steps]

    for i in range(len(steps) - 1):
        assert vals_in[i] <= vals_in[i + 1], "ease_in_cubic must be monotonically increasing"
        assert vals_out[i] >= vals_out[i + 1], "ease_out_cubic must be monotonically decreasing"

    # Middle values curvature
    mid_in = ease_in_cubic(0.5)
    assert mid_in == pytest.approx(1.0 - math.pow(0.5, 3))  # 0.875
    mid_out = ease_out_cubic(0.5)
    assert mid_out == pytest.approx(math.pow(0.5, 3))  # 0.125


def test_magnifier_state_machine_charging_and_active():
    settings = MagnifierSettings(
        enabled=True,
        charge_dwell_s=1.0,
        trigger_dwell_s=1.5,
        zoom_factor=2.2,
        lens_radius_px=130,
        escape_margin_px=60.0,
    )
    controller = MagnifierController(settings)
    stabilizer = FixationStabilizer(radius_px=45.0)

    center = np.array([500.0, 500.0])

    # 1. t=0.0 to t=0.8: Still below charging threshold (< 1.0s) -> IDLE
    stabilizer(center, 0.0)
    status0 = controller.update(stabilizer, center, 0.8)
    assert status0.state == MagnifierState.IDLE
    assert controller.state == MagnifierState.IDLE

    # 2. t=1.2s: Fixation duration is 1.2s (between 1.0s and 1.5s) -> CHARGING
    status1 = controller.update(stabilizer, center, 1.2)
    assert status1.state == MagnifierState.CHARGING
    assert controller.state == MagnifierState.CHARGING
    assert 0.0 < status1.charge_progress < 1.0
    assert status1.center == (500.0, 500.0)

    # 3. t=1.55s: Fixation duration >= 1.5s -> ACTIVE!
    status2 = controller.update(stabilizer, center, 1.55)
    assert status2.state == MagnifierState.ACTIVE
    assert controller.state == MagnifierState.ACTIVE
    assert status2.center == (500.0, 500.0)
    assert status2.zoom_factor == 2.2
    assert status2.charge_progress == 1.0


def test_magnifier_coordinate_remapping():
    settings = MagnifierSettings(zoom_factor=2.0, lens_radius_px=100)
    controller = MagnifierController(settings)

    # Directly activate at center (500, 500)
    controller.center = np.array([500.0, 500.0])
    controller.state = MagnifierState.ACTIVE

    # Looking exactly at lens center -> real target is lens center
    target_center = controller.map_to_real_screen(np.array([500.0, 500.0]))
    assert np.allclose(target_center, [500.0, 500.0])

    # Looking at (550, 500) within the lens: offset is +50 px
    # With zoom 2.0, real target is 500 + 50/2.0 = 525 px
    target_offset = controller.map_to_real_screen(np.array([550.0, 500.0]))
    assert np.allclose(target_offset, [525.0, 500.0])

    # Looking at (460, 480): offset is (-40, -20)
    # Real target is 500 - 40/2 = 480, 500 - 20/2 = 490
    target_neg = controller.map_to_real_screen(np.array([460.0, 480.0]))
    assert np.allclose(target_neg, [480.0, 490.0])


def test_magnifier_escape_behavior():
    settings = MagnifierSettings(
        enabled=True,
        lens_radius_px=100,
        escape_margin_px=50.0,  # escape radius = 150 px
    )
    controller = MagnifierController(settings)
    stabilizer = FixationStabilizer(radius_px=45.0)

    # Activate
    controller.center = np.array([300.0, 300.0])
    controller.state = MagnifierState.ACTIVE

    # Inside lens: dist = 80 px < 150 px -> remains ACTIVE
    inside_pt = np.array([380.0, 300.0])
    status = controller.update(stabilizer, inside_pt, 1.0)
    assert status.state == MagnifierState.ACTIVE

    # Far glance: dist = 250 px > 150 px
    outside_pt = np.array([550.0, 300.0])
    # Brief glance (100 ms) -> stays ACTIVE
    status_brief = controller.update(stabilizer, outside_pt, 1.1)
    assert status_brief.state == MagnifierState.ACTIVE

    # Sustained outside for > 300 ms (t=1.45s - 1.1s = 350 ms) -> resets to IDLE
    status_escaped = controller.update(stabilizer, outside_pt, 1.45)
    assert status_escaped.state == MagnifierState.IDLE
    assert controller.state == MagnifierState.IDLE


def test_magnifier_dismiss_and_reset():
    settings = MagnifierSettings(enabled=True)
    controller = MagnifierController(settings)

    controller.center = np.array([200.0, 200.0])
    controller.state = MagnifierState.ACTIVE

    controller.dismiss()
    assert controller.state == MagnifierState.IDLE
    assert controller.center is None


def test_magnifier_aim_dwell_fired_click():
    settings = MagnifierSettings(
        enabled=True,
        trigger_dwell_s=1.0,
        aim_dwell_s=0.7,
        lens_radius_px=160,
    )
    controller = MagnifierController(settings)
    stabilizer = FixationStabilizer(radius_px=45.0)

    center = np.array([500.0, 500.0])

    # 1. Ativa a lupa em t=1.0s
    stabilizer(center, 0.0)
    status_act = controller.update(stabilizer, center, 1.0)
    assert status_act.state == MagnifierState.ACTIVE
    assert status_act.aim_progress == pytest.approx(0.0)
    assert not status_act.fired_click

    # 2. Metade do aim dwell (t=1.35s -> 0.35s após ativação)
    status_half = controller.update(stabilizer, center, 1.35)
    assert status_half.state == MagnifierState.ACTIVE
    assert status_half.aim_progress == pytest.approx(0.5, abs=0.05)
    assert not status_half.fired_click

    # 3. Completa o aim dwell (t=1.70s -> 0.70s após ativação)
    status_click = controller.update(stabilizer, center, 1.70)
    assert status_click.state == MagnifierState.ACTIVE
    assert status_click.aim_progress == pytest.approx(1.0)
    assert status_click.fired_click

    # 4. Próximo frame (t=1.75s): não repete o disparo
    status_next = controller.update(stabilizer, center, 1.75)
    assert not status_next.fired_click


def test_magnifier_corner_clamping_exact_mapping():
    settings = MagnifierSettings(
        enabled=True,
        trigger_dwell_s=0.5,
        zoom_factor=2.0,
        lens_radius_px=160,
    )
    controller = MagnifierController(settings, screen_size=(1920, 1080))
    stabilizer = FixationStabilizer(radius_px=45.0)

    # Ponto no canto superior esquerdo (20, 20)
    corner = np.array([20.0, 20.0])
    stabilizer(corner, 0.0)
    status = controller.update(stabilizer, corner, 0.6)

    assert status.state == MagnifierState.ACTIVE
    # A janela 320x320 deve ser clampeada na borda (0, 0)
    assert controller.window_pos == (0, 0)
    assert controller.roi_center is not None

    # Mira na posição (160, 160) da tela (que é exatamente o centro da janela da lente)
    gaze_center_lens = np.array([160.0, 160.0])
    target = controller.map_to_real_screen(gaze_center_lens)
    # Deve mapear exatamente para roi_center
    assert target[0] == pytest.approx(controller.roi_center[0])
    assert target[1] == pytest.approx(controller.roi_center[1])


def test_magnifier_wide_saccade_instant_escape():
    settings = MagnifierSettings(
        enabled=True,
        trigger_dwell_s=0.5,
        lens_radius_px=100,
        escape_margin_px=50.0,  # escape radius = 150 px; 2.5x = 375 px
    )
    controller = MagnifierController(settings)
    stabilizer = FixationStabilizer(radius_px=45.0)

    center = np.array([500.0, 500.0])
    stabilizer(center, 0.0)
    controller.update(stabilizer, center, 0.6)
    assert controller.state == MagnifierState.ACTIVE

    # Sacada ampla instantânea para (900, 500), dist = 400 px > 375 px
    saccade_pt = np.array([900.0, 500.0])
    status = controller.update(stabilizer, saccade_pt, 0.65)
    # Deve fechar imediatamente no primeiríssimo frame sem aguardar 250 ms
    assert status.state == MagnifierState.IDLE
    assert controller.state == MagnifierState.IDLE

