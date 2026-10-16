"""Configurações e Fixtures Globais do Pytest para o L.O.R.E.N.A.-DESKTOP.

Fornece fixtures canônicas e padronizadas para:
1. Frames sintéticos (OpenCV/NumPy) sem dependência de hardware.
2. Ambiente de interface gráfica Tkinter headless e isolado com teardown garantido.
3. Mocks de periféricos (Cursor, Câmera, FaceLandmarker).
4. Dados sintéticos de calibração e características oculares (GazeFeatures).
"""

from typing import Generator
from unittest.mock import MagicMock
import numpy as np
import pytest

from lorena.config import Settings
from lorena.domain.models import CalibrationSample, GazeFeatures
from lorena.infra.platform import configure_tkinter_environment
from lorena.services.cursor import Cursor


# ============================================================================
# Fixtures de Imagem e Visão Computacional
# ============================================================================

@pytest.fixture
def synthetic_frame() -> np.ndarray:
    """Retorna um frame BGR sintético de 720p (720, 1280, 3) com luminância média nominal."""
    frame = np.full((720, 1280, 3), 128, dtype=np.uint8)
    return frame


@pytest.fixture
def dummy_face_landmarks() -> np.ndarray:
    """Retorna uma matriz de 478 landmarks normalizados (x, y) representando uma face centralizada."""
    landmarks = np.zeros((478, 2), dtype=np.float64)
    # Define landmarks oculares básicos (íris e cantos em coordenadas simuladas)
    landmarks[:, 0] = 0.5
    landmarks[:, 1] = 0.5
    return landmarks


# ============================================================================
# Fixtures de Modelos de Domínio e Calibração
# ============================================================================

@pytest.fixture
def sample_gaze_features() -> GazeFeatures:
    """Retorna uma instância válida e imutável de GazeFeatures em repouso nominal."""
    return GazeFeatures(
        timestamp=1.0,
        h=0.5,
        v=0.0,
        yaw=0.0,
        pitch=0.0,
        ear_left=0.30,
        ear_right=0.30,
        openness_left=1.0,
        openness_right=1.0,
        quality=1.0,
    )


@pytest.fixture
def sample_calibration_dataset() -> list[CalibrationSample]:
    """Retorna um conjunto sintético balanceado de amostras de calibração em 9 pontos (1080p)."""
    xs = [192.0, 960.0, 1728.0]
    ys = [108.0, 540.0, 972.0]
    samples: list[CalibrationSample] = []
    for ty in ys:
        for tx in xs:
            norm_x = tx / 1920.0
            norm_y = ty / 1080.0
            # Simula mapeamento linear preliminar com leve curvatura
            h = 0.2 + 0.6 * norm_x
            v = -0.3 + 0.6 * norm_y
            for _ in range(5):
                samples.append(
                    CalibrationSample(
                        h=h,
                        v=v,
                        yaw=0.0,
                        pitch=0.0,
                        target_x=tx,
                        target_y=ty,
                    )
                )
    return samples


# ============================================================================
# Fixtures de Periféricos e Serviços Mockados
# ============================================================================

class RecordingMockCursor(Cursor):
    """Cursor gravador em memória para validação determinística de testes."""

    def __init__(self):
        self.pos = np.array([0.0, 0.0], dtype=np.float64)
        self.clicks: list[str] = []

    def move_to(self, pos: np.ndarray) -> None:
        self.pos = np.array(pos, dtype=np.float64)

    def click(self, button: str = "left") -> None:
        self.clicks.append(button)


@pytest.fixture
def mock_cursor() -> RecordingMockCursor:
    """Fornece um cursor mockado seguro e gravador."""
    return RecordingMockCursor()


# ============================================================================
# Fixtures de Interface Gráfica Tkinter (Headless & Isolada)
# ============================================================================

@pytest.fixture
def tk_root() -> Generator:
    """Cria e destrói de forma limpa uma raiz Tkinter em modo headless para testes de UI.
    
    Garante que a janela oculta seja destruída mesmo em caso de erro na asserção,
    prevenindo vazamento de recursos na thread principal.
    """
    configure_tkinter_environment()
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    try:
        yield root
    finally:
        try:
            root.destroy()
        except Exception:
            pass


@pytest.fixture(autouse=True)
def isolate_config_file(tmp_path, monkeypatch) -> None:
    """Garante que a execução de testes nunca altere settings/config.json no repositório."""
    temp_config = tmp_path / "config.json"
    import lorena.paths
    monkeypatch.setattr(lorena.paths, "CONFIG_FILE", temp_config)
    import lorena.ui.main_window
    monkeypatch.setattr(lorena.ui.main_window.paths, "CONFIG_FILE", temp_config)

