import logging
import sys
import threading
import time

import cv2
import numpy as np

from lorena.config import CameraSettings

log = logging.getLogger(__name__)


class CameraError(RuntimeError):
    pass


class Camera:
    def __init__(self, settings: CameraSettings) -> None:
        self._s = settings
        self._cap: cv2.VideoCapture | None = None
        self._exposure_ctrl = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._has_new_frame = threading.Event()
        self._read_failures = 0

    def open(self) -> None:
        if self._cap is not None:
            return
        # No Windows, Media Foundation (MSMF) decodifica por hardware e entrega 30+ FPS,
        # enquanto DirectShow (DSHOW) força YUY2 e cai para 10-14 FPS por limite de largura de banda USB.
        backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
        cap = None
        for backend in backends:
            c = cv2.VideoCapture(self._s.index, backend)
            if c.isOpened():
                # No Windows MSMF, as propriedades PRECISAM ser definidas antes da primeira
                # leitura de quadro, caso contrário o MSMF falha ao tentar reconfigurar o stream.
                try:
                    c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                except Exception:
                    pass
                c.set(cv2.CAP_PROP_FRAME_WIDTH, self._s.width)
                c.set(cv2.CAP_PROP_FRAME_HEIGHT, self._s.height)
                c.set(cv2.CAP_PROP_FPS, self._s.fps)
                c.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                try:
                    ok, test_frame = c.read()
                    if ok and test_frame is not None:
                        cap = c
                        break
                except Exception:
                    pass
            c.release()

        # Fallback defensivo: se a câmera rejeitou 720p/MJPG, tenta abrir na resolução padrão do driver
        if cap is None:
            for backend in backends:
                c = cv2.VideoCapture(self._s.index, backend)
                if c.isOpened():
                    try:
                        ok, test_frame = c.read()
                        if ok and test_frame is not None:
                            cap = c
                            break
                    except Exception:
                        pass
                c.release()

        if cap is None or not cap.isOpened():
            raise CameraError(f"Não foi possível abrir a câmera {self._s.index}.")

        real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        real_fps = float(cap.get(cv2.CAP_PROP_FPS))
        log.info("Câmera %d aberta: %dx%d a %.1f FPS (solicitado %dx%d a %d FPS)",
                 self._s.index, real_w, real_h, real_fps, self._s.width, self._s.height, self._s.fps)
        if real_w < 1280 or real_h < 720:
            log.warning("Câmera operando abaixo de 720p (%dx%d); a precisão do rastreio nos cantos pode ser reduzida.",
                        real_w, real_h)

        if getattr(self._s, "exposure_control", "off") == "auto":
            try:
                from lorena.vision.exposure import ExposureController
                self._exposure_ctrl = ExposureController(cap, target_luma=getattr(self._s, "target_face_luma", 120.0))
                self._exposure_ctrl.start()
            except Exception as exc:
                log.warning("Não foi possível inicializar controle de exposição: %s", exc)

        self._cap = cap
        self._stop_event.clear()
        self._has_new_frame.clear()
        self._latest_frame = None
        self._read_failures = 0
        self._thread = threading.Thread(target=self._capture_worker, name="camera-capture", daemon=True)
        self._thread.start()

    def _capture_worker(self) -> None:
        """Loop contínuo em thread dedicada para esgotar buffers de hardware sem bloquear a visão."""
        while not self._stop_event.is_set():
            cap = self._cap
            if cap is None:
                break
            try:
                ok, frame = cap.read()
            except Exception as exc:
                log.debug("Exceção na captura do frame: %s", exc)
                ok, frame = False, None

            if ok and frame is not None:
                with self._frame_lock:
                    self._latest_frame = frame.copy()
                    self._read_failures = 0
                self._has_new_frame.set()
            else:
                with self._frame_lock:
                    self._read_failures += 1
                time.sleep(0.005)

    def update_exposure(self, face_luma: float, t: float) -> None:
        if self._exposure_ctrl is not None:
            self._exposure_ctrl.update(face_luma, t)

    def read(self, timeout: float = 0.5) -> np.ndarray | None:
        if self._cap is None:
            raise CameraError("Câmera não aberta.")

        # Fallback síncrono para testes ou situações sem thread iniciada:
        if self._thread is None or not self._thread.is_alive():
            try:
                ok, frame = self._cap.read()
                return frame if ok and frame is not None else None
            except Exception:
                return None

        # Aguarda chegada assíncrona do novo quadro capturado pela thread de hardware:
        if self._has_new_frame.wait(timeout=timeout):
            self._has_new_frame.clear()
            with self._frame_lock:
                return self._latest_frame

        with self._frame_lock:
            if self._read_failures >= 30:
                return None
            return self._latest_frame

    def release(self) -> None:
        if self._thread is not None:
            self._stop_event.set()
            self._has_new_frame.set()
            if self._thread is not threading.current_thread():
                self._thread.join(timeout=1.5)
            self._thread = None

        if self._exposure_ctrl is not None:
            try:
                self._exposure_ctrl.restore()
            except Exception:
                pass
            self._exposure_ctrl = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        with self._frame_lock:
            self._latest_frame = None

    def set_index(self, index: int) -> None:
        if self._cap is not None:
            raise CameraError("Não é possível alterar a fonte com a câmera aberta.")
        self._s.index = index

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def list_available_cameras(max_tested: int = 4) -> list[tuple[int, str]]:
    """Detecta dispositivos de captura disponíveis (índice, descrição).
    
    Analisa os quadros de teste para diferenciar câmeras coloridas (RGB) de
    sensores infravermelhos (ex: Windows Hello, imagem preta/ruído), priorizando
    as câmeras de cor no topo da lista.
    """
    import contextlib
    import os

    color_cams: list[tuple[int, str]] = []
    other_cams: list[tuple[int, str]] = []
    backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW] if sys.platform == "win32" else [cv2.CAP_ANY]

    # Suprime avisos de baixo nível do OpenCV ao testar índices não conectados
    @contextlib.contextmanager
    def _suppress_c_stderr():
        if "pytest" in sys.modules:
            yield
            return
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            stderr_fd = 2
            saved = os.dup(stderr_fd)
            os.dup2(devnull, stderr_fd)
            os.close(devnull)
            try:
                yield
            finally:
                os.dup2(saved, stderr_fd)
                os.close(saved)
        except Exception:
            yield

    with _suppress_c_stderr():
        for idx in range(max_tested):
            cap = None
            for backend in backends:
                try:
                    c = cv2.VideoCapture(idx, backend)
                    if c.isOpened():
                        c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        ok, test_frame = c.read()
                        if ok and test_frame is not None:
                            cap = c
                            break
                    c.release()
                except Exception:
                    pass

            if cap is not None and cap.isOpened():
                try:
                    ok, test_frame = cap.read()
                    if ok and test_frame is not None:
                        mean_val = float(test_frame.mean())
                        is_color = False
                        if test_frame.ndim == 3 and test_frame.shape[2] == 3:
                            diff_rg = float(np.abs(test_frame[:, :, 0].astype(float) - test_frame[:, :, 1].astype(float)).mean())
                            diff_rb = float(np.abs(test_frame[:, :, 0].astype(float) - test_frame[:, :, 2].astype(float)).mean())
                            is_color = (diff_rg > 1.0 or diff_rb > 1.0)

                        # Exclui sensores infravermelhos (ex: Windows Hello, imagem preta/ruído)
                        # e fontes sem iluminação utilizável para rastreamento ocular
                        if mean_val > 8.0 and (is_color or mean_val > 15.0):
                            color_cams.append((idx, f"Câmera {idx}"))
                except Exception:
                    pass
                finally:
                    cap.release()

    if not color_cams:
        color_cams.append((0, "Câmera 0"))
    return color_cams

