import time
from lorena.config import Settings
from lorena.vision.camera import Camera
from lorena.services.tracking_pipeline import TrackingPipeline
from lorena.vision.face_detector import FaceLandmarkDetector
from lorena.vision.gaze_features import GazeFeatureExtractor, OpennessModel
from lorena import paths

settings = Settings.load(paths.CONFIG_FILE)
print(f"Loaded camera settings: index={settings.camera.index}")

cam = Camera(settings.camera)
try:
    print("Opening camera...")
    cam.open()
    print("Camera opened successfully!")
    for i in range(5):
        frame = cam.read()
        print(f"Frame {i}: {'OK' if frame is not None else 'None'}")
        time.sleep(0.05)
finally:
    print("Releasing camera...")
    cam.release()
    print("Camera released.")
