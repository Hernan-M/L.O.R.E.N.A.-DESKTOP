import time
import cv2

print("--- Testing DSHOW with MJPG ---")
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
cap.set(cv2.CAP_PROP_FOURCC, fourcc)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

times = []
for _ in range(5): cap.read()
t0 = time.perf_counter()
for _ in range(30):
    ok, f = cap.read()
    if ok: times.append(time.perf_counter())
t1 = time.perf_counter()
cap.release()
print(f"DSHOW with MJPG Delivered FPS: {len(times)/(t1-t0):.2f} FPS")

print("\n--- Testing MSMF with MJPG ---")
cap2 = cv2.VideoCapture(1, cv2.CAP_MSMF)
cap2.set(cv2.CAP_PROP_FOURCC, fourcc)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap2.set(cv2.CAP_PROP_FPS, 30)

times2 = []
for _ in range(5): cap2.read()
t0 = time.perf_counter()
for _ in range(30):
    ok, f = cap2.read()
    if ok: times2.append(time.perf_counter())
t1 = time.perf_counter()
cap2.release()
print(f"MSMF with MJPG Delivered FPS: {len(times2)/(t1-t0):.2f} FPS")
