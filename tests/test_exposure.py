import cv2
import pytest

from lorena.vision.exposure import ExposureController


class MockCapture:
    def __init__(self, accept_manual: bool = True):
        self.accept_manual = accept_manual
        self.props = {
            cv2.CAP_PROP_AUTO_EXPOSURE: 0.75,
            cv2.CAP_PROP_EXPOSURE: -6.0,
            cv2.CAP_PROP_GAIN: 10.0,
        }

    def get(self, prop: int) -> float:
        return self.props.get(prop, 0.0)

    def set(self, prop: int, value: float) -> bool:
        if prop == cv2.CAP_PROP_AUTO_EXPOSURE and not self.accept_manual:
            return False
        self.props[prop] = value
        return True


def test_exposure_controller_start_success():
    cap = MockCapture(accept_manual=True)
    ctrl = ExposureController(cap, target_luma=120.0)
    started = ctrl.start()
    assert started is True
    assert ctrl.active is True
    assert cap.props[cv2.CAP_PROP_AUTO_EXPOSURE] == 0.25


def test_exposure_controller_fallback_when_rejected():
    cap = MockCapture(accept_manual=False)
    ctrl = ExposureController(cap, target_luma=120.0)
    started = ctrl.start()
    assert started is False
    assert ctrl.active is False


def test_exposure_controller_adjusts_towards_target():
    cap = MockCapture(accept_manual=True)
    ctrl = ExposureController(cap, target_luma=120.0, deadband=10.0, min_interval=0.1)
    ctrl.start()

    init_exp = cap.props[cv2.CAP_PROP_EXPOSURE]
    # Dark face (60 luma < 120 target) -> should increase exposure
    ctrl.update(face_luma=60.0, now=1.0)
    assert cap.props[cv2.CAP_PROP_EXPOSURE] > init_exp

    # Calling immediately before min_interval should NOT adjust
    exp_before = cap.props[cv2.CAP_PROP_EXPOSURE]
    ctrl.update(face_luma=60.0, now=1.05)
    assert cap.props[cv2.CAP_PROP_EXPOSURE] == exp_before

    # Restoring resets properties
    ctrl.restore()
    assert ctrl.active is False
