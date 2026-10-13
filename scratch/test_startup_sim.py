import tkinter as tk
import time
from lorena.config import Settings
from lorena.services.tracking_pipeline import TrackingPipeline
from lorena.services.cursor import Win32Cursor
from lorena.infra.calibration_repo import CalibrationRepository
from lorena.ui.main_window import AppServices, MainWindow
from lorena.vision.camera import Camera
from lorena.vision.face_detector import FaceLandmarkDetector
from lorena.vision.gaze_features import GazeFeatureExtractor, OpennessModel
from lorena import paths

root = tk.Tk()
settings = Settings.load(paths.CONFIG_FILE)
print("Config camera index:", settings.camera.index)

pipeline = TrackingPipeline(
    camera=Camera(settings.camera),
    detector_factory=lambda: FaceLandmarkDetector(paths.FACE_LANDMARKER_MODEL, settings.detector),
    extractor=GazeFeatureExtractor(OpennessModel(settings.blink)),
)
services = AppServices(
    settings=settings,
    screen=(1920, 1080),
    pipeline=pipeline,
    repository=CalibrationRepository(paths.CALIBRATION_DIR),
    cursor=Win32Cursor(),
)
win = MainWindow(root, services)
print("Available cameras found by MainWindow:", win._cameras)
print("Combobox values:", win._cb_source["values"] if win._cb_source else None)
print("Combobox current:", win._cb_source.current() if win._cb_source else None)
print("Pipeline camera index:", pipeline._camera._s.index)

# Let root update for 1 second
for _ in range(20):
    root.update()
    time.sleep(0.05)

print("Preview running:", win._preview.is_running if win._preview else False)
print("Latest rgb frame in preview:", win._preview._latest_rgb is not None if win._preview else False)

win._on_close()
print("Clean shutdown.")
