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


def test_list_available_cameras_classification(monkeypatch):
    import numpy as np

    class MockCap:
        def __init__(self, idx):
            self.idx = idx
            self._opened = idx in (0, 1)

        def isOpened(self):
            return self._opened

        def set(self, prop, val):
            return True

        def read(self):
            if self.idx == 0:
                # Monocromático / baixa luz
                frame = np.full((10, 10, 3), 4, dtype=np.uint8)
                return True, frame
            elif self.idx == 1:
                # RGB / Colorido
                frame = np.zeros((10, 10, 3), dtype=np.uint8)
                frame[:, :, 0] = 200
                frame[:, :, 1] = 50
                frame[:, :, 2] = 20
                return True, frame
            return False, None

        def release(self):
            self._opened = False

    monkeypatch.setattr("cv2.VideoCapture", lambda idx, *args: MockCap(idx))

    cams = list_available_cameras(max_tested=3)
    assert len(cams) == 2
    # Câmera RGB (1) deve vir antes de Câmera IR/Pouca luz (0)
    assert cams[0][0] == 1
    assert "RGB" in cams[0][1]
    assert cams[1][0] == 0
    assert "IR" in cams[1][1] or "Monocromático" in cams[1][1]


def test_camera_open_with_retries(monkeypatch):
    import numpy as np
    cam_cfg = CameraSettings(index=0)
    cam = Camera(cam_cfg)

    call_count = 0
    fake_frame = np.full((10, 10, 3), 100, dtype=np.uint8)

    class MockFlakyCap:
        def __init__(self):
            nonlocal call_count
            call_count += 1
            # Falha nas primeiras 2 tentativas, sucesso na 3ª
            self._opened = (call_count >= 3)

        def isOpened(self):
            return self._opened

        def set(self, prop, val):
            return True

        def get(self, prop):
            return 30.0

        def read(self):
            return True, fake_frame

        def release(self):
            self._opened = False

    monkeypatch.setattr("cv2.VideoCapture", lambda *args: MockFlakyCap())
    monkeypatch.setattr("time.sleep", lambda s: None)

    cam.open()
    assert cam._cap is not None
    assert cam._cap.isOpened()
    cam.release()


def test_capture_worker_resilient_to_cv2_error(monkeypatch):
    import cv2
    import numpy as np
    import time

    cam_cfg = CameraSettings(index=0)
    cam = Camera(cam_cfg)

    fake_frame = np.full((10, 10, 3), 100, dtype=np.uint8)
    error_raised = False

    class MockCapWithCv2Error:
        def isOpened(self):
            return True

        def set(self, prop, val):
            return True

        def get(self, prop):
            return 30.0

        def read(self):
            nonlocal error_raised
            if not error_raised:
                error_raised = True
                raise cv2.error("assertion failed: _step >= minstep")
            return True, fake_frame

        def release(self):
            pass

    monkeypatch.setattr("cv2.VideoCapture", lambda *args: MockCapWithCv2Error())

    cam.open()
    # Aguarda o worker processar a exceção e em seguida o frame válido
    frame = cam.read(timeout=0.5)
    assert frame is not None
    assert np.array_equal(frame, fake_frame)
    cam.release()


def test_list_available_cameras_bright_ir_sensor_classified_as_ir(monkeypatch):
    """Garante que sensores monocromáticos/IR mesmo com alta exposição (luma > 15) não sejam falsamente categorizados como RGB."""
    import numpy as np

    class MockCap:
        def __init__(self, idx):
            self.idx = idx
            self._opened = (idx == 0)

        def isOpened(self):
            return self._opened

        def set(self, prop, val):
            return True

        def read(self):
            # Alta exposição (luma = 80), mas perfeitamente monocromático (canais B=G=R)
            frame = np.full((10, 10, 3), 80, dtype=np.uint8)
            return True, frame

        def release(self):
            self._opened = False

    monkeypatch.setattr("cv2.VideoCapture", lambda idx, *args: MockCap(idx))

    cams = list_available_cameras(max_tested=1)
    assert len(cams) == 1
    assert cams[0][0] == 0
    # Deve ser classificado como IR e NÃO como RGB
    assert "RGB" not in cams[0][1]
    assert "IR" in cams[0][1] or "Monocromático" in cams[0][1]


def test_list_available_cameras_early_break(monkeypatch):
    """Garante que a sondagem é interrompida no primeiro índice que falhar, sem testar índices superiores inexistentes."""
    import numpy as np

    tested_indices = []

    class MockCap:
        def __init__(self, idx):
            tested_indices.append(idx)
            # Câmera 0 abre, Câmera 1 falha
            self._opened = (idx == 0)

        def isOpened(self):
            return self._opened

        def set(self, prop, val):
            return True

        def read(self):
            frame = np.zeros((10, 10, 3), dtype=np.uint8)
            frame[:, :, 0] = 200
            frame[:, :, 1] = 50
            return True, frame

        def release(self):
            pass

    monkeypatch.setattr("cv2.VideoCapture", lambda idx, *args: MockCap(idx))

    cams = list_available_cameras(max_tested=4)
    assert len(cams) == 1
    assert cams[0][0] == 0
    # Verificou índice 0, falhou no 1, e NÃO testou 2 nem 3
    assert 0 in tested_indices
    assert 1 in tested_indices
    assert 2 not in tested_indices
    assert 3 not in tested_indices


