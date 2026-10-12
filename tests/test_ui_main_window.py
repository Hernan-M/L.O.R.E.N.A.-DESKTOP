from unittest.mock import MagicMock
import pytest

from lorena.config import Settings
from lorena.infra.platform import configure_tkinter_environment
from lorena.ui.main_window import AppServices, MainWindow


def test_main_window_camera_select(tmp_path):
    configure_tkinter_environment()
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()

    settings = Settings()
    mock_pipeline = MagicMock()
    mock_repo = MagicMock()
    mock_repo.load.return_value = None
    mock_cursor = MagicMock()

    services = AppServices(
        settings=settings,
        screen=(1920, 1080),
        pipeline=mock_pipeline,
        repository=mock_repo,
        cursor=mock_cursor,
    )

    try:
        win = MainWindow(root, services)
        assert win._cb_source is not None
        assert len(win._cameras) > 0

        # Verifica que o combobox reflete a lista de câmeras
        values = list(win._cb_source["values"])
        assert len(values) == len(win._cameras)

        # Testa desativação em busy
        win._set_busy(True)
        assert str(win._cb_source.cget("state")) == "disabled"
        win._set_busy(False)
        assert str(win._cb_source.cget("state")) == "readonly"

        assert win._preview is not None

        # Simula troca de câmera
        if len(win._cameras) > 1:
            win._cb_source.current(1)
            win._on_source_changed()
            target_idx = win._cameras[1][0]
            assert settings.camera.index == target_idx
            mock_pipeline.set_camera_index.assert_called_with(target_idx)
            assert win._preview.camera_index == target_idx

        win._on_close()
    finally:
        try:
            root.destroy()
        except Exception:
            pass
