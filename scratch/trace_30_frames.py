import time
import cv2
import numpy as np
from lorena import paths
from lorena.config import Settings
from lorena.vision.camera import Camera
from lorena.vision.face_detector import FaceLandmarkDetector

settings = Settings()
settings.camera.index = 1
cam = Camera(settings.camera)
cam.open()
detector = FaceLandmarkDetector(paths.FACE_LANDMARKER_MODEL, settings.detector)

print("Tracing 30 frames with individual detect times:")
for i in range(30):
    f = cam.read()
    if f is None: continue
    t0 = time.perf_counter()
    face = detector.detect(f, time.monotonic())
    t1 = time.perf_counter()
    dt = (t1 - t0) * 1000.0
    print(f"Frame {i:02d}: detect={dt:.1f}ms (face={'YES' if face else 'NO'})")

cam.release()
detector.close()
