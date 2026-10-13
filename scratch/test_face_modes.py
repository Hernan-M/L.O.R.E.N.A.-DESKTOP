import time
import cv2
import numpy as np
from lorena import paths
from lorena.config import DetectorSettings
from lorena.vision.face_detector import FaceLandmarkDetector

# Let's create a realistic face image or load one to see MediaPipe when face IS detected
detector = FaceLandmarkDetector(paths.FACE_LANDMARKER_MODEL, DetectorSettings(use_clahe=False))

# Draw a more realistic face pattern that triggers MediaPipe Face Detector
face_img = np.full((720, 1280, 3), 150, dtype=np.uint8)
# Head
cv2.ellipse(face_img, (640, 360), (180, 240), 0, 0, 360, (210, 180, 160), -1)
# Eyes
cv2.circle(face_img, (580, 320), 22, (255, 255, 255), -1)
cv2.circle(face_img, (700, 320), 22, (255, 255, 255), -1)
cv2.circle(face_img, (580, 320), 10, (50, 40, 30), -1)
cv2.circle(face_img, (700, 320), 10, (50, 40, 30), -1)
# Nose
cv2.line(face_img, (640, 340), (640, 400), (180, 140, 120), 4)
# Mouth
cv2.ellipse(face_img, (640, 440), (40, 15), 0, 0, 180, (160, 80, 80), -1)

# Check detection
res = detector.detect(face_img, 0.033)
print("Realistic synthetic face detected:", res is not None)

# Also test Camera 1 with actual person in front of webcam:
cap = cv2.VideoCapture(1, cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("\nTesting 20 frames on Camera 1 with face detection status:")
for i in range(20):
    ok, frame = cap.read()
    if not ok: continue
    t0 = time.perf_counter()
    face = detector.detect(frame, time.monotonic())
    t1 = time.perf_counter()
    print(f"Frame {i}: detected={'YES' if face is not None else 'NO'}, time={(t1-t0)*1000:.1f}ms")

cap.release()
detector.close()
