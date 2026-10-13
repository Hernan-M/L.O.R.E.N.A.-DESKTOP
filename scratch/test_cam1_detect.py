import time
import cv2
from lorena import paths
from lorena.config import DetectorSettings
from lorena.vision.face_detector import FaceLandmarkDetector

cap = cv2.VideoCapture(1, cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

detector = FaceLandmarkDetector(paths.FACE_LANDMARKER_MODEL, DetectorSettings(use_clahe=True))

print("Capturing 10 frames from Camera 1 with FaceLandmarkDetector:")
for i in range(10):
    t0 = time.perf_counter()
    ok, frame = cap.read()
    t1 = time.perf_counter()
    face = detector.detect(frame, time.monotonic())
    t2 = time.perf_counter()
    stats = detector.last_stats
    print(f"Frame {i}: read={(t1-t0)*1000:.1f}ms, detect={(t2-t1)*1000:.1f}ms, face={'YES' if face else 'NO'}, stats={stats}")

cap.release()
detector.close()
