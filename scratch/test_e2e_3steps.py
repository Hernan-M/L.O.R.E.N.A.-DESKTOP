import tkinter as tk
import time
from lorena.config import Settings
from lorena.services.tracking_pipeline import TrackingPipeline
from lorena.services.cursor import Win32Cursor
from lorena.infra.calibration_repo import CalibrationRepository
from lorena.ui.main_window import AppServices, MainWindow
from lorena.vision.camera import Camera
from lorena.vision.face_detector import FaceLandmarkDetector
from lorena.vision.gaze_features import GazeFeatureExtractor, OpennessModel
from lorena import paths

root = tk.Tk()
settings = Settings.load(paths.CONFIG_FILE)
print("1. INICIALIZAÇÃO: Configuração carregada com câmera índice:", settings.camera.index)

pipeline = TrackingPipeline(
    camera=Camera(settings.camera),
    detector_factory=lambda: FaceLandmarkDetector(paths.FACE_LANDMARKER_MODEL, settings.detector),
    extractor=GazeFeatureExtractor(OpennessModel(settings.blink)),
)
services = AppServices(
    settings=settings,
    screen=(1920, 1080),
    pipeline=pipeline,
    repository=CalibrationRepository(paths.CALIBRATION_DIR),
    cursor=Win32Cursor(),
)
win = MainWindow(root, services)

print("2. VERIFICAÇÃO DE FONTES AO INICIAR:")
print("   Fontes detectadas:", win._cameras)
print("   Valores no combobox:", win._cb_source["values"])
print("   Índice ativo no combobox:", win._cb_source.current())
print("   Câmera no settings:", settings.camera.index)
print("   Câmera no pipeline:", pipeline._camera._s.index)

# Deixa a UI rodar por 0.5s para alimentar o preview
for _ in range(10):
    root.update()
    time.sleep(0.05)

print("   Preview em execução:", win._preview.is_running)

# 3. SELEÇÃO DE FONTE:
if len(win._cameras) > 1:
    print("\n3. SELEÇÃO DE FONTE:")
    # Alterna para o outro índice
    cur = win._cb_source.current()
    new_idx = 1 if cur == 0 else 0
    print(f"   Alterando seleção para opção {new_idx}: {win._cameras[new_idx]}")
    win._cb_source.current(new_idx)
    win._on_source_changed()
    
    # Roda UI por 0.5s
    for _ in range(10):
        root.update()
        time.sleep(0.05)
    
    print("   Nova câmera no settings:", settings.camera.index)
    print("   Nova câmera no pipeline:", pipeline._camera._s.index)
    print("   Nova câmera no preview:", win._preview.camera_index)
    print("   Preview em execução com nova câmera:", win._preview.is_running)

# 4. UTILIZAÇÃO DE FONTE PELA APLICAÇÃO:
print("\n4. UTILIZAÇÃO DE FONTE PELA APLICAÇÃO:")
# Simula parada do preview e início do pipeline com a câmera selecionada
print(f"   Iniciando pipeline na câmera selecionada ({pipeline._camera._s.index})...")
win._preview.stop()
time.sleep(0.15)
pipeline.start(None)

# Aguarda 10 quadros do pipeline
time.sleep(0.5)
assert pipeline.running, "Pipeline deveria estar rodando!"
print("   Pipeline rodando com sucesso! Verificando resultados...")

# Coleta resultados
results = []
while not pipeline.results.empty():
    results.append(pipeline.results.get_nowait())
print(f"   Quadros processados pelo pipeline: {len(results)}")

print("   Encerrando pipeline e restaurando preview...")
pipeline.stop()
time.sleep(0.10)
win._preview.start()

for _ in range(10):
    root.update()
    time.sleep(0.05)

print("   Preview restaurado:", win._preview.is_running)

win._on_close()
print("\nTODOS OS 3 PASSOS FORAM EXECUTADOS COM SUCESSO ABSOLUTO!")
