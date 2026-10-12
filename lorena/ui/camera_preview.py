"""Componente de preview de vídeo simples da câmera para o menu inicial."""
from __future__ import annotations

import contextlib
import logging
import os
import sys
import threading
import time
import tkinter as tk
from tkinter.ttk import Frame, Label
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from lorena.ui import theme

log = logging.getLogger(__name__)

DEFAULT_WIDTH = 160
DEFAULT_HEIGHT = 120
POLL_INTERVAL_MS = 60  # ~16 FPS de atualização na interface


@contextlib.contextmanager
def _suppress_c_stderr():
    """Suprime mensagens de baixo nível de drivers no Windows."""
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


class CameraPreview(Frame):
    """Widget de preview compacto e não-bloqueante para a webcam."""

    def __init__(
        self,
        parent: tk.Widget,
        camera_index: int = 0,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        style: str = "Card.TFrame",
    ) -> None:
        super().__init__(parent, style=style)
        self._camera_index = camera_index
        self._width = width
        self._height = height

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest_rgb: Optional[np.ndarray] = None
        self._photo_image: Optional[ImageTk.PhotoImage] = None
        self._poll_id: Optional[str] = None

        self._border_frame = tk.Frame(
            self,
            bg=theme.ACCENT,
            padx=1,
            pady=1,
        )
        self._border_frame.pack()

        self._label = tk.Label(
            self._border_frame,
            bg=theme.SURFACE,
            width=self._width,
            height=self._height,
        )
        self._label.pack()

        self._show_placeholder("Conectando...")
        self.start()

    @property
    def camera_index(self) -> int:
        return self._camera_index

    def set_camera_index(self, index: int) -> None:
        """Altera a câmera exibida no preview."""
        if self._camera_index == index and self.is_running:
            return
        self.stop()
        self._camera_index = index
        self._show_placeholder("Conectando...")
        self.start()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Inicia a captura em thread secundária e o polling na UI."""
        if self.is_running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, name="camera-preview", daemon=True)
        self._thread.start()
        self._schedule_poll()

    def stop(self) -> None:
        """Encerra a captura e libera imediatamente o dispositivo da câmera."""
        self._stop_event.set()
        if self._poll_id is not None:
            self.after_cancel(self._poll_id)
            self._poll_id = None
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.5)
        self._thread = None
        with self._lock:
            self._latest_rgb = None
        self._show_placeholder("Câmera pausada")

    def _show_placeholder(self, text: str) -> None:
        """Gera imagem estática com mensagem para manter as dimensões estáveis."""
        img = np.full((self._height, self._width, 3), (17, 38, 56), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.38
        thickness = 1
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        tx = max(4, (self._width - tw) // 2)
        ty = max(th + 4, (self._height + th) // 2)
        cv2.putText(img, text, (tx, ty), font, font_scale, (143, 166, 190), thickness, cv2.LINE_AA)

        try:
            photo = ImageTk.PhotoImage(image=Image.fromarray(img))
            self._photo_image = photo
            self._label.configure(image=photo)
        except Exception:
            pass

    def _worker(self) -> None:
        backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
        cap = None

        with _suppress_c_stderr():
            for backend in backends:
                if self._stop_event.is_set():
                    return
                cap = cv2.VideoCapture(self._camera_index, backend)
                if cap.isOpened():
                    break
                cap.release()
                cap = None

        if cap is None or not cap.isOpened():
            log.info("Preview: câmera %d indisponível ou em uso.", self._camera_index)
            if not self._stop_event.is_set():
                self.after(0, lambda: self._show_placeholder("Sem sinal"))
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        target_delay = 1.0 / 15.0
        try:
            while not self._stop_event.is_set():
                t0 = time.monotonic()
                ok, frame = cap.read()
                if ok and frame is not None:
                    # Espelhado horizontalmente para experiência amigável de webcam
                    mirrored = cv2.flip(frame, 1)
                    resized = cv2.resize(mirrored, (self._width, self._height))
                    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    with self._lock:
                        self._latest_rgb = rgb
                else:
                    time.sleep(0.04)

                elapsed = time.monotonic() - t0
                wait_time = max(0.005, target_delay - elapsed)
                self._stop_event.wait(wait_time)
        finally:
            cap.release()

    def _schedule_poll(self) -> None:
        if not self._stop_event.is_set():
            self._poll_id = self.after(POLL_INTERVAL_MS, self._poll_frame)

    def _poll_frame(self) -> None:
        if self._stop_event.is_set():
            return
        rgb = None
        with self._lock:
            if self._latest_rgb is not None:
                rgb = self._latest_rgb
                self._latest_rgb = None

        if rgb is not None:
            try:
                photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
                self._photo_image = photo
                self._label.configure(image=photo)
            except Exception:
                pass

        self._schedule_poll()
