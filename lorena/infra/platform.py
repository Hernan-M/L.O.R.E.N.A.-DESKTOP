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
            tcl_candidates = list(base.rglob("init.tcl"))
            if tcl_candidates:
                tcl_path = tcl_candidates[0].parent
                os.environ["TCL_LIBRARY"] = str(tcl_path).replace("\\", "/")
                parent = tcl_path.parent
                tk_candidates = [p for p in parent.glob("tk8*") if p.is_dir()]
                if tk_candidates:
                    os.environ["TK_LIBRARY"] = str(tk_candidates[0]).replace("\\", "/")
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


def screen_physical_mm() -> tuple[float, float]:
    """Tamanho físico (largura, altura) da tela principal em milímetros."""
    if IS_WINDOWS:
        try:
            u = ctypes.windll.user32
            g = ctypes.windll.gdi32
            hdc = u.GetDC(0)
            w_mm = float(g.GetDeviceCaps(hdc, 4))  # HORZSIZE
            h_mm = float(g.GetDeviceCaps(hdc, 6))  # VERTSIZE
            u.ReleaseDC(0, hdc)
            if w_mm > 50 and h_mm > 50:
                return w_mm, h_mm
        except Exception:
            pass
    # Fallback assumindo 96 DPI padrão
    w_px, h_px = screen_size()
    return float(w_px * 25.4 / 96.0), float(h_px * 25.4 / 96.0)


def screen_pixel_pitch_mm() -> float:
    """Tamanho médio de um pixel na tela em milímetros."""
    w_px, h_px = screen_size()
    w_mm, h_mm = screen_physical_mm()
    if w_px > 0 and h_px > 0 and w_mm > 0 and h_mm > 0:
        pitch_x = w_mm / w_px
        pitch_y = h_mm / h_px
        avg = (pitch_x + pitch_y) / 2.0
        if 0.10 <= avg <= 0.50:
            return float(avg)
    return float(25.4 / 96.0)


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
