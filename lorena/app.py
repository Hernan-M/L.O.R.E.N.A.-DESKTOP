"""Composition root: o único lugar que instancia classes concretas."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from lorena.infra.platform import configure_tkinter_environment, enable_dpi_awareness, screen_size

configure_tkinter_environment()

import tkinter as tk

from lorena import paths
from lorena.config import Settings
from lorena.infra.calibration_repo import CalibrationRepository
from lorena.services.cursor import Win32Cursor
from lorena.services.tracking_pipeline import TrackingPipeline
from lorena.ui.main_window import AppServices, MainWindow
from lorena.vision.camera import Camera
from lorena.vision.face_detector import FaceLandmarkDetector
from lorena.vision.gaze_features import GazeFeatureExtractor, OpennessModel

log = logging.getLogger(__name__)



def configure_logging() -> None:
    paths.LOG_DIR.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(paths.LOG_DIR / "lorena.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s",
        handlers=[handler, logging.StreamHandler()],
    )


def main() -> None:
    configure_logging()
    enable_dpi_awareness()  # antes de qualquer janela
    settings = Settings.load(paths.CONFIG_FILE)
    if not paths.CONFIG_FILE.exists():
        settings.save(paths.CONFIG_FILE)  # gera o arquivo editável com os padrões
    log.info("Iniciando L.O.R.E.N.A. — Configuração carregada (câmera índice=%d)", settings.camera.index)

    pipeline = TrackingPipeline(
        camera=Camera(settings.camera),
        detector_factory=lambda: FaceLandmarkDetector(paths.FACE_LANDMARKER_MODEL, settings.detector),
        extractor=GazeFeatureExtractor(OpennessModel(settings.blink)),
    )
    services = AppServices(
        settings=settings,
        screen=screen_size(),
        pipeline=pipeline,
        repository=CalibrationRepository(paths.CALIBRATION_DIR),
        cursor=Win32Cursor(),
    )
    root = tk.Tk()
    MainWindow(root, services)
    root.mainloop()


if __name__ == "__main__":
    main()
