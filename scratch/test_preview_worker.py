import cv2
import sys

# Test what CameraPreview._worker does directly for camera 1
backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
cap = None
for backend in backends:
    print(f"Trying backend {backend} for cam 1...")
    try:
        c = cv2.VideoCapture(1, backend)
        print("  isOpened:", c.isOpened())
        if c.isOpened():
            c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, test_frame = c.read()
            print("  read ok:", ok, "frame is not None:", test_frame is not None)
            if ok and test_frame is not None:
                cap = c
                break
        c.release()
    except Exception as e:
        print("  Exception:", e)

if cap:
    print("Cap successfully acquired! Reading 5 frames:")
    for i in range(5):
        ok, f = cap.read()
        print(f"  Frame {i}: ok={ok}, shape={f.shape if f is not None else None}")
    cap.release()
else:
    print("Cap FAILED to acquire!")
