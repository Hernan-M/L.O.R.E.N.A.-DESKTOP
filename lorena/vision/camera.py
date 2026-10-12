"""Captura da webcam."""
from __future__ import annotations

import sys

import cv2
import numpy as np

from lorena.config import CameraSettings


class CameraError(RuntimeError):
    pass


class Camera:
    def __init__(self, settings: CameraSettings) -> None:
        self._s = settings
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        if self._cap is not None:
            return
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
        cap = None
        for backend in backends:
            cap = cv2.VideoCapture(self._s.index, backend)
            if cap.isOpened():
                break
            cap.release()
            cap = None
        if cap is None or not cap.isOpened():
            raise CameraError(f"Não foi possível abrir a câmera {self._s.index}.")
        # Mais resolução = mais pixels por olho = landmarks mais estáveis.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._s.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._s.height)
        cap.set(cv2.CAP_PROP_FPS, self._s.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap

    def read(self) -> np.ndarray | None:
        if self._cap is None:
            raise CameraError("Câmera não aberta.")
        ok, frame = self._cap.read()
        return frame if ok and frame is not None else None

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

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
    """Detecta dispositivos de captura disponíveis (índice, descrição)."""
    import contextlib
    import os

    available: list[tuple[int, str]] = []
    backend = cv2.CAP_MSMF if sys.platform == "win32" else cv2.CAP_ANY

    # Suprime avisos de baixo nível do OpenCV ao testar índices não conectados
    @contextlib.contextmanager
    def _suppress_c_stderr():
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
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened() and backend != cv2.CAP_ANY:
                cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                available.append((idx, f"Câmera {idx}"))
                cap.release()

    if not available:
        available.append((0, "Câmera 0"))
    return available
