import numpy as np
import pytest

from lorena.infra.screen_capture import ScreenCapture


def test_screen_capture_shape_and_type():
    capture = ScreenCapture()
    # Capture standard 100x100 ROI at center of 1920x1080 display
    roi = capture.capture_roi(center_x=960, center_y=540, width=100, height=100, screen_size=(1920, 1080))
    assert isinstance(roi, np.ndarray)
    assert roi.shape == (100, 100, 3)
    assert roi.dtype == np.uint8


def test_screen_capture_edge_clamping():
    capture = ScreenCapture()
    # Top-left corner (0, 0)
    roi_top_left = capture.capture_roi(center_x=0, center_y=0, width=120, height=120, screen_size=(1920, 1080))
    assert roi_top_left.shape == (120, 120, 3)

    # Bottom-right corner (1920, 1080)
    roi_bottom_right = capture.capture_roi(center_x=1920, center_y=1080, width=120, height=120, screen_size=(1920, 1080))
    assert roi_bottom_right.shape == (120, 120, 3)

    # Small screen (1366, 768)
    roi_small = capture.capture_roi(center_x=1360, center_y=760, width=80, height=80, screen_size=(1366, 768))
    assert roi_small.shape == (80, 80, 3)
