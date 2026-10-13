import cv2
import numpy as np
import sys
import contextlib
import os

@contextlib.contextmanager
def _suppress_c_stderr():
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        stderr_fd = 2
        saved = os.dup(stderr_fd)
        os.dup2(devnull, stderr_fd)
        os.close(devnull)
        try:
            yield
        finally:
            os.dup2(saved, stderr_fd)
            os.close(saved)
    except Exception:
        yield

def detect_cameras(max_tested=4):
    color_cams = []
    other_cams = []
    backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW] if sys.platform == "win32" else [cv2.CAP_ANY]

    with _suppress_c_stderr():
        for idx in range(max_tested):
            cap = None
            for backend in backends:
                try:
                    c = cv2.VideoCapture(idx, backend)
                    if c.isOpened():
                        c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        ok, test_frame = c.read()
                        if ok and test_frame is not None:
                            cap = c
                            break
                    c.release()
                except Exception:
                    pass

            if cap is not None and cap.isOpened():
                try:
                    ok, test_frame = cap.read()
                    if ok and test_frame is not None:
                        mean_val = float(test_frame.mean())
                        is_color = False
                        if test_frame.ndim == 3 and test_frame.shape[2] == 3:
                            diff_rg = float(np.abs(test_frame[:, :, 0].astype(float) - test_frame[:, :, 1].astype(float)).mean())
                            diff_rb = float(np.abs(test_frame[:, :, 0].astype(float) - test_frame[:, :, 2].astype(float)).mean())
                            is_color = (diff_rg > 1.0 or diff_rb > 1.0)

                        if mean_val > 8.0 and (is_color or mean_val > 15.0):
                            label = f"Câmera {idx} (RGB / Colorida)"
                            color_cams.append((idx, label))
                        else:
                            label = f"Câmera {idx} (Sensor IR / Pouca luz)"
                            other_cams.append((idx, label))
                except Exception:
                    pass
                finally:
                    cap.release()

    res = color_cams + other_cams
    if not res:
        res = [(0, "Câmera 0")]
    return res

cams = detect_cameras()
print("Detected cameras:", cams)
