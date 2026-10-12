"""Integração com o sistema operacional (Windows via ctypes, com fallback inerte)."""
from __future__ import annotations

import ctypes
import logging
import sys

import os
from pathlib import Path

log = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == "win32"
VK_ESCAPE = 0x1B


def configure_tkinter_environment() -> None:
    """Configura automaticamente TCL_LIBRARY e TK_LIBRARY no Windows quando o Python
    está num venv ou quando as bibliotecas Tcl não estão no caminho padrão."""
    if not IS_WINDOWS:
        return
    if "TCL_LIBRARY" in os.environ and "TK_LIBRARY" in os.environ:
        return

    localapp = os.environ.get("LOCALAPPDATA", "")
    search_dirs = [
        Path(localapp) / "Programs" / "Python",
        Path("C:/Python313"),
        Path(sys.base_prefix),
        Path(sys.prefix),
    ]

    for base in search_dirs:
        if not base.exists():
            continue
        try:
            tcl_dirs = list(base.glob("**/tcl/tcl8*")) + list(base.glob("tcl/tcl8*"))
            if tcl_dirs:
                tcl_path = tcl_dirs[0]
                os.environ.setdefault("TCL_LIBRARY", str(tcl_path))
                tk_path = tcl_path.parent / f"tk{tcl_path.name.replace('tcl', '')}"
                if tk_path.exists():
                    os.environ.setdefault("TK_LIBRARY", str(tk_path))
                break
        except Exception:
            continue


configure_tkinter_environment()


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
