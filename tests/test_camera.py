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


def test_threaded_camera_capture_and_release(monkeypatch):
    import numpy as np
    import time
    cam_cfg = CameraSettings(index=0)
    cam = Camera(cam_cfg)

    fake_frame = np.ones((100, 100, 3), dtype=np.uint8) * 120
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, fake_frame)
    mock_cap.get.side_effect = lambda prop: 100.0 if prop in (3, 4) else 30.0

    # Evita abrir dispositivo real
    monkeypatch.setattr("cv2.VideoCapture", lambda *args, **kwargs: mock_cap)

    cam.open()
    assert cam._thread is not None
    assert cam._thread.is_alive()

    # Leitura assíncrona desacoplada
    frame = cam.read(timeout=0.2)
    assert frame is not None
    assert np.array_equal(frame, fake_frame)

    # Parada e liberação de thread
    cam.release()
    assert cam._thread is None
    assert cam._cap is None
    assert cam._latest_frame is None


def test_camera_synchronous_fallback_when_thread_idle():
    import numpy as np
    cam_cfg = CameraSettings(index=0)
    cam = Camera(cam_cfg)

    fake_frame = np.full((10, 10, 3), 42, dtype=np.uint8)
    mock_cap = MagicMock()
    mock_cap.read.return_value = (True, fake_frame)
    cam._cap = mock_cap

    # Sem iniciar thread, read() usa fallback síncrono
    frame = cam.read()
    assert frame is not None
    assert np.array_equal(frame, fake_frame)
    mock_cap.read.assert_called_once()

