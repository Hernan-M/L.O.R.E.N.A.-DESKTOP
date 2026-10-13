import time
import cv2
import numpy as np
from lorena import paths
from lorena.config import DetectorSettings
from lorena.vision.face_detector import FaceLandmarkDetector

# Test with MSMF + MJPG
cap = cv2.VideoCapture(1, cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

detector = FaceLandmarkDetector(paths.FACE_LANDMARKER_MODEL, DetectorSettings(use_clahe=True))

durations = []
for i in range(30):
    t0 = time.perf_counter()
    ok, frame = cap.read()
    if not ok: continue
    face = detector.detect(frame, time.monotonic())
    t1 = time.perf_counter()
    durations.append((t1 - t0) * 1000.0)

cap.release()
detector.close()

avg_ms = np.mean(durations[5:])
fps = 1000.0 / avg_ms
print(f"MSMF + MJPG Full Cycle: avg={avg_ms:.2f} ms -> {fps:.2f} FPS")
