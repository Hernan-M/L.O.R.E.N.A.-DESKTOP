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
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self._s.index, backend)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self._s.index)
        if not cap.isOpened():
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

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.release()
