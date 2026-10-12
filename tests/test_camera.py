import pytest
from unittest.mock import MagicMock

from lorena.config import CameraSettings
from lorena.vision.camera import Camera, CameraError, list_available_cameras
from lorena.services.tracking_pipeline import TrackingPipeline


def test_camera_set_index_idle():
    cam_cfg = CameraSettings(index=0)
    cam = Camera(cam_cfg)
    assert cam_cfg.index == 0

    cam.set_index(1)
    assert cam_cfg.index == 1


def test_camera_set_index_while_open():
    cam_cfg = CameraSettings(index=0)
    cam = Camera(cam_cfg)
    cam._cap = MagicMock()  # Simula câmera aberta

    with pytest.raises(CameraError, match="Não é possível alterar a fonte"):
        cam.set_index(2)


def test_tracking_pipeline_set_camera_index():
    cam_cfg = CameraSettings(index=0)
    cam = Camera(cam_cfg)
    mock_detector_factory = MagicMock()
    mock_extractor = MagicMock()

    pipeline = TrackingPipeline(
        camera=cam,
        detector_factory=mock_detector_factory,
        extractor=mock_extractor,
    )

    pipeline.set_camera_index(2)
    assert cam_cfg.index == 2


def test_list_available_cameras_structure():
    # Testa com max_tested pequeno para ser rápido
    cams = list_available_cameras(max_tested=2)
    assert isinstance(cams, list)
    assert len(cams) >= 1
    for idx, name in cams:
        assert isinstance(idx, int)
        assert isinstance(name, str)
