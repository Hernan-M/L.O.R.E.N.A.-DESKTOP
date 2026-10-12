"""Integração com o sistema operacional (Windows via ctypes, com fallback inerte)."""
from __future__ import annotations

import ctypes
import logging
import sys

log = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == "win32"
VK_ESCAPE = 0x1B


def enable_dpi_awareness() -> None:
    """Sem isso, com escala 125%/150% as coordenadas do cursor ficam desalinhadas.
    Deve ser chamada ANTES de criar qualquer janela."""
    if not IS_WINDOWS:
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            log.warning("Não foi possível ativar DPI awareness.")


def screen_size() -> tuple[int, int]:
    if IS_WINDOWS:
        u = ctypes.windll.user32
        return int(u.GetSystemMetrics(0)), int(u.GetSystemMetrics(1))
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    size = root.winfo_screenwidth(), root.winfo_screenheight()
    root.destroy()
    return size


class HotkeyListener:
    """Detecta tecla pressionada globalmente (mesmo sem janela focada)."""

    def __init__(self, vk: int = VK_ESCAPE) -> None:
        self._vk = vk
        self._was_down = False

    def pressed(self) -> bool:
        """Verdadeiro uma única vez por pressionamento (detecção de borda)."""
        if not IS_WINDOWS:
            return False
        down = bool(ctypes.windll.user32.GetAsyncKeyState(self._vk) & 0x8000)
        fired = down and not self._was_down
        self._was_down = down
        return fired
