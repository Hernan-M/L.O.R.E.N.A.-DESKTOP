import tkinter as tk
from unittest.mock import MagicMock

from lorena.infra.platform import configure_tkinter_environment
from lorena.ui.camera_preview import CameraPreview


def test_camera_preview_widget_lifecycle():
    configure_tkinter_environment()
    root = tk.Tk()
    root.withdraw()

    try:
        preview = CameraPreview(root, camera_index=0, width=160, height=120)
        assert preview.camera_index == 0
        assert preview._width == 160
        assert preview._height == 120
        assert preview._label is not None

        # Testa troca de índice
        preview.set_camera_index(1)
        assert preview.camera_index == 1

        # Testa parada limpa
        preview.stop()
        assert not preview.is_running
        assert preview._poll_id is None

        # Testa parada idempotente
        preview.stop()
    finally:
        root.destroy()
