"""Testes para o composition root (lorena.app)."""
import logging
from unittest.mock import MagicMock
import pytest

import lorena.app as app
from lorena.config import Settings


def test_app_logger_defined():
    """Garante que o logger de módulo 'log' está definido e configurado."""
    assert hasattr(app, "log")
    assert isinstance(app.log, logging.Logger)


def test_app_main_composition_root(monkeypatch, tmp_path):
    """Garante que main() inicializa todos os serviços sem NameError ou falhas de composição."""
    # Isola caminhos de configuração e logs
    monkeypatch.setattr(app.paths, "CONFIG_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(app.paths, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(app.paths, "CALIBRATION_DIR", tmp_path / "calibrations")

    # Mocka chamadas com efeitos colaterais
    monkeypatch.setattr(app, "enable_dpi_awareness", lambda: None)
    monkeypatch.setattr(app, "screen_size", lambda: (1920, 1080))
    monkeypatch.setattr(app, "Win32Cursor", MagicMock)

    # Mocka MainWindow e o Tkinter para isolamento total
    mock_main_window = MagicMock()
    mock_tk_instance = MagicMock()
    monkeypatch.setattr("tkinter.Tk", lambda: mock_tk_instance)
    monkeypatch.setattr(app, "MainWindow", mock_main_window)

    # Executa main
    app.main()


    # Verifica que MainWindow recebeu os serviços corretos
    mock_main_window.assert_called_once()
    services = mock_main_window.call_args[0][1]
    assert services.screen == (1920, 1080)
    assert isinstance(services.settings, Settings)
