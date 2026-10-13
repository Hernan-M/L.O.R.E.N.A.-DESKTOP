import time
import cv2
import numpy as np
from lorena.vision.illumination import IlluminationNormalizer

cap = cv2.VideoCapture(1, cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

norm = IlluminationNormalizer()

print("Testing IlluminationNormalizer on real Camera 1 frames:")
for i in range(10):
    ok, frame = cap.read()
    if not ok: continue
    t0 = time.perf_counter()
    out, stats = norm.process(frame, None)
    t1 = time.perf_counter()
    print(f"Frame {i}: took {(t1-t0)*1000:.2f} ms (stats: {stats})")

cap.release()
