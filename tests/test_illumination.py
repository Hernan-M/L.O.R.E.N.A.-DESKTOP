import cv2
import numpy as np
import pytest

from lorena.vision.illumination import IlluminationNormalizer


def test_illumination_roi_backlight():
    # Backlit scene: bright background (220 luma), dark face ROI (50 luma)
    img = np.full((200, 200, 3), 220, dtype=np.uint8)
    roi = (50, 50, 150, 150)
    img[50:150, 50:150] = 50

    normalizer = IlluminationNormalizer(clahe_on=90.0, clahe_off=120.0)
    out, stats = normalizer.process(img, roi)

    # Face ROI was detected as dark (50), triggering adaptation
    assert stats.face_luma < 90.0
    assert stats.clahe is True
    # The output image in ROI should have higher average intensity/contrast
    roi_out = cv2.cvtColor(out[50:150, 50:150], cv2.COLOR_BGR2GRAY)
    assert float(roi_out.mean()) > 50.0


def test_illumination_hysteresis_no_chatter():
    normalizer = IlluminationNormalizer(clahe_on=80.0, clahe_off=100.0)
    roi = (10, 10, 90, 90)

    # 1. Dark image triggers CLAHE
    dark = np.full((100, 100, 3), 70, dtype=np.uint8)
    _, s1 = normalizer.process(dark, roi)
    assert s1.clahe is True

    # 2. Borderline image (90 luma) stays active due to hysteresis
    mid = np.full((100, 100, 3), 90, dtype=np.uint8)
    _, s2 = normalizer.process(mid, roi)
    assert s2.clahe is True

    # 3. Bright image (> 100) deactivates CLAHE
    bright = np.full((100, 100, 3), 110, dtype=np.uint8)
    _, s3 = normalizer.process(bright, roi)
    assert s3.clahe is False

    # 4. Moving back to 90 does NOT re-trigger because clahe_on is 80
    _, s4 = normalizer.process(mid, roi)
    assert s4.clahe is False


def test_illumination_continuous_gamma():
    normalizer = IlluminationNormalizer(target=0.5, ema=0.5)
    img = np.full((100, 100, 3), 80, dtype=np.uint8)

    _, s1 = normalizer.process(img)
    _, s2 = normalizer.process(img)
    # Gamma smoothly adjusts without sudden discontinuity
    assert s1.gamma != 1.0
    assert abs(s2.gamma - s1.gamma) < 0.5


def test_illumination_fast_mean_std_and_global_gamma():
    # Testa normalização suave e contínua sem criar descontinuidades ou degraus no rosto
    img = np.full((100, 100, 3), 60, dtype=np.uint8)
    roi = (20, 20, 80, 80)

    normalizer = IlluminationNormalizer(target=0.7, ema=1.0, deadband=0.01)
    out, stats = normalizer.process(img, roi)

    assert stats.gamma != 1.0
    assert stats.face_luma == 60.0
    # O frame é processado continuamente sem degrau entre dentro e fora do ROI
    assert out.shape == img.shape
    assert not np.all(out == 60)


def test_illumination_dynamic_bypass_stable_light():
    # Luz estável (ex: luminância 120, dentro de [85, 170])
    img = np.full((100, 100, 3), 120, dtype=np.uint8)
    roi = (20, 20, 80, 80)
    normalizer = IlluminationNormalizer(stable_range=(85.0, 170.0))
    out, stats = normalizer.process(img, roi)

    # Deve efetuar bypass: frame original repassado intacto, gamma 1.0, sem CLAHE
    assert out is img
    assert stats.face_luma == 120.0
    assert stats.gamma == 1.0
    assert stats.clahe is False

