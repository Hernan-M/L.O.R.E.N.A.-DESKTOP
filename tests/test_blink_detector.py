import pytest

from lorena.config import BlinkSettings
from lorena.domain.models import GestureEvent
from lorena.vision.blink_detector import BlinkDetector


def test_natural_bilateral_blink_is_ignored():
    settings = BlinkSettings()
    detector = BlinkDetector(settings)

    # Initial open
    t = 0.0
    assert detector.update(1.0, 1.0, t) is None
    assert not detector.freeze

    # Both eyes close (natural blink)
    t = 0.05
    assert detector.update(0.1, 0.1, t) is None
    assert detector.state == "both"

    # Both eyes reopen at 150ms
    t = 0.15
    event = detector.update(1.0, 1.0, t)
    assert event is None  # Natural blink does NOT click
    assert detector.state == "open"


def test_left_wink_triggers_click():
    settings = BlinkSettings(min_wink_s=0.20, max_click_s=1.00)
    detector = BlinkDetector(settings)

    t = 0.0
    assert detector.update(1.0, 1.0, t) is None

    # Left eye closes, right stays open
    t = 0.10
    assert detector.update(0.15, 0.9, t) is None
    assert detector.state == "wink"
    assert detector.freeze

    # After 300ms, left eye reopens
    t = 0.40
    event = detector.update(0.85, 0.9, t)
    assert event == GestureEvent.CLICK
    assert detector.state == "refractory"

    # Refractory period prevents immediate repeat
    t = 0.50
    assert detector.update(0.15, 0.9, t) is None

    # Reopens after refractory (0.4s later)
    t = 0.85
    assert detector.update(0.9, 0.9, t) is None
    assert detector.state == "open"


def test_long_wink_triggers_right_click():
    settings = BlinkSettings(min_wink_s=0.20, max_click_s=1.00, max_long_wink_s=2.50)
    detector = BlinkDetector(settings)

    t = 0.0
    detector.update(1.0, 1.0, t)

    # Left eye closes
    t = 0.10
    detector.update(0.15, 0.9, t)

    # Reopens after 1.4 seconds (long wink)
    t = 1.50
    event = detector.update(0.9, 0.9, t)
    assert event == GestureEvent.RIGHT_CLICK


def test_both_eyes_hold_toggles_pause():
    settings = BlinkSettings(pause_hold_s=1.20)
    detector = BlinkDetector(settings)

    t = 0.0
    detector.update(1.0, 1.0, t)

    # Both eyes close
    t = 0.10
    detector.update(0.1, 0.1, t)

    # Held closed for 1.5s
    t = 1.60
    event = detector.update(0.9, 0.9, t)
    assert event == GestureEvent.TOGGLE_PAUSE
