import time
import threading
import cv2

backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]

for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
    name = "DSHOW" if backend == cv2.CAP_DSHOW else ("MSMF" if backend == cv2.CAP_MSMF else "ANY")
    print(f"\n--- Testing Backend {name} ---")
    cap = cv2.VideoCapture(1, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Measure inside a worker thread
    read_durations = []
    def worker():
        for _ in range(10):
            t0 = time.perf_counter()
            ok, f = cap.read()
            t1 = time.perf_counter()
            read_durations.append((t1 - t0) * 1000.0)

    th = threading.Thread(target=worker)
    th.start()
    th.join()
    cap.release()
    print(f"Read durations (ms): {[round(d, 1) for d in read_durations]}")
    print(f"Average: {sum(read_durations)/len(read_durations):.1f} ms")
