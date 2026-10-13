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
        # Priorizamos MSMF em todas as tentativas antes de permitir qualquer fallback para DSHOW.
        cap = None
        max_attempts = 3

        if sys.platform == "win32":
            # 1. Tentar prioritariamente Media Foundation (MSMF) em todas as tentativas com backoff
            for attempt in range(max_attempts):
                c = cv2.VideoCapture(self._s.index, cv2.CAP_MSMF)
                if c.isOpened():
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
                    except (cv2.error, Exception):
                        pass
                c.release()

                if attempt < max_attempts - 1:
                    log.debug("Aguardando liberação do MSMF para câmera %d (tentativa %d/%d)...",
                              self._s.index, attempt + 1, max_attempts)
                    time.sleep(0.15)

            # 2. Fallback defensivo para DirectShow ou genérico se MSMF falhar todas as tentativas
            if cap is None:
                log.warning("MSMF indisponível para câmera %d; tentando backends secundários (DSHOW/ANY)...", self._s.index)
                for backend in [cv2.CAP_DSHOW, cv2.CAP_ANY]:
                    c = cv2.VideoCapture(self._s.index, backend)
                    if c.isOpened():
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
                        except (cv2.error, Exception):
                            pass
                    c.release()
        else:
            # Plataformas não-Windows (Linux / macOS)
            for attempt in range(max_attempts):
                c = cv2.VideoCapture(self._s.index, cv2.CAP_ANY)
                if c.isOpened():
                    c.set(cv2.CAP_PROP_FRAME_WIDTH, self._s.width)
                    c.set(cv2.CAP_PROP_FRAME_HEIGHT, self._s.height)
                    c.set(cv2.CAP_PROP_FPS, self._s.fps)
                    c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    try:
                        ok, test_frame = c.read()
                        if ok and test_frame is not None:
                            cap = c
                            break
                    except (cv2.error, Exception):
                        pass
                c.release()
                if attempt < max_attempts - 1:
                    time.sleep(0.15)

        if cap is None or not cap.isOpened():
            raise CameraError(f"Não foi possível abrir a câmera {self._s.index}.")

        real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        real_fps = float(cap.get(cv2.CAP_PROP_FPS))
        backend_name = cap.getBackendName() if hasattr(cap, "getBackendName") else "Desconhecido"
        log.info("Câmera %d aberta via backend %s: %dx%d a %.1f FPS (solicitado %dx%d a %d FPS)",
                 self._s.index, backend_name, real_w, real_h, real_fps, self._s.width, self._s.height, self._s.fps)
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
            except (cv2.error, Exception) as exc:
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
        log.info("Câmera: índice alterado para %d", index)

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def list_available_cameras(max_tested: int = 4) -> list[tuple[int, str]]:
    """Detecta dispositivos de captura disponíveis (índice, descrição).
    
    Analisa os quadros de teste para diferenciar câmeras coloridas (RGB) de
    sensores infravermelhos (ex: Windows Hello, imagem preta/ruído), priorizando
    as câmeras de cor no topo da lista sem descartar outras fontes funcionais.
    Interrompe a busca logo na primeira falha contígua para não bloquear a UI.
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
            initial_frame = None
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
                            initial_frame = test_frame
                            break
                    c.release()
                except Exception:
                    pass

            if cap is None or initial_frame is None:
                # Índices no Windows e Linux são atribuídos de forma contígua a partir de 0.
                # Se este índice falhou em todos os backends, interrompe imediatamente para não bloquear a UI.
                break

            try:
                mean_val = float(initial_frame.mean())
                is_color = False
                if initial_frame.ndim == 3 and initial_frame.shape[2] == 3:
                    # Subamostragem 8x8 para variância instantânea (< 0.05 ms) sem alocações pesadas
                    sample = initial_frame[::8, ::8, :].astype(np.int16)
                    # Padrão BGR do OpenCV: canal 0=Blue, canal 1=Green, canal 2=Red
                    diff_bg = float(np.abs(sample[:, :, 0] - sample[:, :, 1]).mean())
                    diff_br = float(np.abs(sample[:, :, 0] - sample[:, :, 2]).mean())
                    diff_gr = float(np.abs(sample[:, :, 1] - sample[:, :, 2]).mean())
                    is_color = max(diff_bg, diff_br, diff_gr) > 1.5

                if mean_val > 8.0 and is_color:
                    color_cams.append((idx, f"Câmera {idx} (RGB / Colorida)"))
                elif mean_val <= 8.0:
                    other_cams.append((idx, f"Câmera {idx} (Sensor IR ou Tampa Fechada)"))
                else:
                    other_cams.append((idx, f"Câmera {idx} (Sensor Monocromático)"))
            except Exception:
                pass
            finally:
                cap.release()

    res = color_cams + other_cams
    if not res:
        res.append((0, "Câmera 0"))
    return res

