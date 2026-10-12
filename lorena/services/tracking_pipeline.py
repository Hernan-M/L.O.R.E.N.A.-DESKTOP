"""Pipeline de visão: UMA thread lê a câmera, detecta e publica resultados.

Substitui as variáveis globais compartilhadas entre threads. A UI consome
``results`` (fila limitada) no próprio thread do Tk via ``after``.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from typing import Callable, Protocol

from lorena.domain.models import FrameResult, GazeFeatures, SessionStatus
from lorena.infra.platform import HotkeyListener
from lorena.vision.camera import Camera
from lorena.vision.gaze_features import GazeFeatureExtractor

log = logging.getLogger(__name__)


class FrameConsumer(Protocol):
    def on_frame(self, f: GazeFeatures | None, t: float) -> SessionStatus: ...


class Detector(Protocol):
    def detect(self, frame, timestamp: float): ...
    def close(self) -> None: ...


class TrackingPipeline:
    MAX_READ_FAILURES = 30

    def __init__(self, camera: Camera, detector_factory: Callable[[], Detector], extractor: GazeFeatureExtractor,
                 results_maxsize: int = 64) -> None:
        self._camera = camera
        self._detector_factory = detector_factory
        self.extractor = extractor
        self.results: queue.Queue[FrameResult] = queue.Queue(maxsize=results_maxsize)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, consumer: FrameConsumer | None = None, stop_on_escape: bool = False) -> None:
        if self.running:
            raise RuntimeError("Pipeline já está em execução.")
        self._drain()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(consumer, stop_on_escape), name="vision", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout)
        self._thread = None

    def _drain(self) -> None:
        while True:
            try:
                self.results.get_nowait()
            except queue.Empty:
                return

    def _publish(self, result: FrameResult) -> None:
        try:
            self.results.put_nowait(result)
        except queue.Full:  # UI atrasada: descarta o mais antigo
            try:
                self.results.get_nowait()
            except queue.Empty:
                pass
            self.results.put_nowait(result)

    def _run(self, consumer: FrameConsumer | None, stop_on_escape: bool) -> None:
        hotkey = HotkeyListener()
        frame_times: deque[float] = deque(maxlen=30)
        detector = None
        failures = 0
        try:
            self._camera.open()
            detector = self._detector_factory()
            while not self._stop.is_set():
                frame = self._camera.read()
                if frame is None:
                    failures += 1
                    if failures >= self.MAX_READ_FAILURES:
                        raise RuntimeError("Falha ao capturar imagem da webcam.")
                    time.sleep(0.01)
                    continue
                failures = 0
                t = time.monotonic()
                frame_times.append(t)
                fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0.0

                features = self.extractor.extract(detector.detect(frame, t))
                status = consumer.on_frame(features, t) if consumer is not None else None

                if stop_on_escape and hotkey.pressed():
                    self._publish(FrameResult(t, features, fps, status, stopped_by_user=True))
                    break
                self._publish(FrameResult(t, features, fps, status))
        except Exception as exc:  # noqa: BLE001 - reportado à UI
            log.exception("Erro no pipeline de visão")
            self._publish(FrameResult(time.monotonic(), None, 0.0, error=str(exc)))
        finally:
            self._camera.release()
            if detector is not None:
                detector.close()
