import cv2
import time

for delay in [0.0, 0.05, 0.1, 0.2]:
    print(f"\n--- Testing delay {delay}s between release and open ---")
    c1 = cv2.VideoCapture(1, cv2.CAP_MSMF)
    opened1 = c1.isOpened()
    ok1, f1 = c1.read() if opened1 else (False, None)
    c1.release()
    del c1
    if delay > 0:
        time.sleep(delay)
    
    c2 = cv2.VideoCapture(1, cv2.CAP_MSMF)
    opened2 = c2.isOpened()
    ok2, f2 = c2.read() if opened2 else (False, None)
    print(f"c1 ok={ok1}, c2 opened={opened2}, c2 ok={ok2}")
    c2.release()
    del c2
