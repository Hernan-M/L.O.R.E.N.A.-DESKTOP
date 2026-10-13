import time
import cv2
import sys
from lorena.config import CameraSettings
from lorena.vision.camera import Camera, list_available_cameras

print("--- 1. VERIFICAÇÃO DE FONTES DISPONÍVEIS ---")
cams = list_available_cameras(max_tested=4)
print(f"Fontes detectadas ({len(cams)}):", cams)
assert len(cams) >= 1

print("\n--- 2. SELEÇÃO DE FONTE ---")
selected_idx = cams[0][0]
print(f"Fonte selecionada: {selected_idx} ({cams[0][1]})")

cfg = CameraSettings(index=selected_idx, width=1280, height=720, fps=30)
cam = Camera(cfg)

print("\n--- 3. UTILIZAÇÃO DE FONTE PELA APLICAÇÃO ---")
print("Abrindo câmera selecionada no Camera...")
t0 = time.monotonic()
cam.open()
print(f"Câmera aberta em {time.monotonic() - t0:.3f}s!")

frames_read = 0
for i in range(10):
    f = cam.read(timeout=0.5)
    if f is not None:
        frames_read += 1
    time.sleep(0.03)

print(f"Quadros lidos com sucesso: {frames_read}/10")
assert frames_read >= 8

print("Liberando câmera...")
cam.release()
print("Câmera liberada com sucesso!")

# Now test switching to second camera if available
if len(cams) > 1:
    second_idx = cams[1][0]
    print(f"\n--- Testando troca para segunda fonte: índice {second_idx} ({cams[1][1]}) ---")
    cfg2 = CameraSettings(index=second_idx, width=1280, height=720, fps=30)
    cam2 = Camera(cfg2)
    cam2.open()
    f2 = cam2.read(timeout=0.5)
    print("Quadro da segunda fonte:", "OK" if f2 is not None else "None")
    cam2.release()
    print("Segunda fonte liberada com sucesso!")

print("\nFluxo completo verificado com sucesso!")
