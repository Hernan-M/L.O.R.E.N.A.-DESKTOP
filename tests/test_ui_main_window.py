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
            # Seleciona uma câmera diferente da selecionada inicialmente
            current_cb = win._cb_source.current()
            new_cb = 0 if current_cb != 0 else 1
            win._cb_source.current(new_cb)
            win._on_source_changed()
            target_idx = win._cameras[new_cb][0]
            assert settings.camera.index == target_idx
            mock_pipeline.set_camera_index.assert_called_with(target_idx)
            assert win._preview.camera_index == target_idx

        win._on_close()
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def test_main_window_play_and_stop_lifecycle():
    configure_tkinter_environment()
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()

    settings = Settings()
    mock_pipeline = MagicMock()
    mock_pipeline.running = False
    mock_repo = MagicMock()
    fake_profile = MagicMock()
    fake_profile.screen_size = [1920, 1080]
    fake_profile.report.loo_mean_error_px = 35.0
    fake_profile.calibration_luma = 120.0
    fake_profile.open_ear_left = 0.28
    fake_profile.open_ear_right = 0.28
    mock_repo.load.return_value = fake_profile
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
        assert win._preview is not None
        mock_preview = MagicMock()
        win._preview = mock_preview

        # Dispara Play (rastreio)
        win._on_play()
        mock_preview.stop.assert_called_once()
        mock_pipeline.start.assert_called_once()

        # Encerra rastreio
        win._stop_tracking()
        mock_pipeline.stop.assert_called_once()
        mock_preview.start.assert_called_once()

        win._on_close()
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def test_main_window_empty_cameras_handling(monkeypatch):
    configure_tkinter_environment()
    import tkinter as tk

    monkeypatch.setattr("lorena.ui.main_window.list_available_cameras", lambda: [])

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
        assert len(win._cameras) == 0
        assert str(win._cb_source.cget("state")) == "disabled"
        assert list(win._cb_source["values"]) == ["Nenhuma câmera"]
        assert str(win._btn_play.cget("state")) == "disabled"
        assert str(win._btn_cal.cget("state")) == "disabled"
        win._on_close()
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def test_main_window_refresh_cameras(monkeypatch):
    configure_tkinter_environment()
    import tkinter as tk

    cams_detected = [(0, "Câmera 0 (RGB / Colorida)")]
    monkeypatch.setattr("lorena.ui.main_window.list_available_cameras", lambda: cams_detected)

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
        assert len(win._cameras) == 1
        assert win._btn_refresh is not None

        # Simula conexão de nova câmera em tempo de execução
        cams_detected = [(0, "Câmera 0 (RGB / Colorida)"), (1, "Câmera 1: ACER HD (RGB)")]
        win._on_refresh_cameras()

        assert len(win._cameras) == 2
        assert list(win._cb_source["values"]) == ["Câmera 0 (RGB / Colorida)", "Câmera 1: ACER HD (RGB)"]
        assert win._cb_source.current() == 0  # Preserva câmera 0 selecionada

        win._on_close()
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def test_main_window_auto_promotes_rgb_camera_over_default_ir(monkeypatch):
    configure_tkinter_environment()
    import tkinter as tk

    # Câmera 0 é IR, Câmera 1 é RGB
    cams_detected = [(1, "Câmera 1 (RGB / Colorida)"), (0, "Câmera 0 (Sensor IR ou Tampa Fechada)")]
    monkeypatch.setattr("lorena.ui.main_window.list_available_cameras", lambda: cams_detected)

    root = tk.Tk()
    root.withdraw()

    # Configuração padrão tem index: 0 (IR)
    settings = Settings()
    settings.camera.index = 0
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
        # Deve ter auto-promovido para Câmera 1 (RGB)
        assert win._svc.settings.camera.index == 1
        assert win._cb_source.current() == 0  # Índice 0 na lista é Câmera 1 (RGB)
        win._on_close()
    finally:
        try:
            root.destroy()
        except Exception:
            pass
