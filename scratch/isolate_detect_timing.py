import time
import cv2
import numpy as np
from lorena import paths
from lorena.config import DetectorSettings
from lorena.vision.face_detector import FaceLandmarkDetector

cap = cv2.VideoCapture(1, cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

detector = FaceLandmarkDetector(paths.FACE_LANDMARKER_MODEL, DetectorSettings(use_clahe=True))

print("Isolating timing inside FaceLandmarkDetector.detect:")
for i in range(10):
    ok, frame = cap.read()
    if not ok: continue
    t_start = time.perf_counter()

    t0 = time.perf_counter()
    processed_bgr, stats = detector._illum.process(frame, detector._roi)
    t1 = time.perf_counter()

    rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
    t2 = time.perf_counter()

    image = detector._mp.Image(image_format=detector._mp.ImageFormat.SRGB, data=rgb)
    result = detector._landmarker.detect_for_video(image, int(time.monotonic()*1000))
    t3 = time.perf_counter()

    print(f"Frame {i}: illum={(t1-t0)*1000:.2f}ms, cvtColor={(t2-t1)*1000:.2f}ms, mediapipe={(t3-t2)*1000:.2f}ms, TOTAL={(t3-t_start)*1000:.2f}ms, clahe_active={stats.clahe}")

cap.release()
detector.close()
