import time
import cv2
from lorena.config import Settings
from lorena.vision.camera import Camera
from lorena import paths

# Simulate what CameraPreview does:
cap_preview = cv2.VideoCapture(1, cv2.CAP_MSMF)
print("Preview opened:", cap_preview.isOpened())
ok, f = cap_preview.read()
print("Preview frame:", ok)

# Now what happens if MainWindow calls preview.stop(), then immediately pipeline.start() -> cam.open()?
# Suppose preview is released:
print("Simulating preview stop...")
cap_preview.release()
del cap_preview

# Now open camera in Camera:
settings = Settings.load(paths.CONFIG_FILE)
cam = Camera(settings.camera)
try:
    print("Opening pipeline camera immediately...")
    cam.open()
    print("Pipeline camera opened successfully!")
    f = cam.read()
    print("Pipeline frame:", "OK" if f is not None else "None")
finally:
    cam.release()
