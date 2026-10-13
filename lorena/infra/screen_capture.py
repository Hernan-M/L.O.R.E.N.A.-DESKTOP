"""Captura ultrarrápida de regiões da tela (ROI) via Win32 GDI BitBlt em sub-milissegundos."""
from __future__ import annotations

import ctypes
import logging
import sys
from ctypes import byref, c_void_p, sizeof

import numpy as np

log = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == "win32"
SRCCOPY = 0x00CC0020
CAPTUREBLT = 0x40000000
DIB_RGB_COLORS = 0


if IS_WINDOWS:
    from ctypes.wintypes import DWORD, LONG, WORD

    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ("biSize", DWORD),
            ("biWidth", LONG),
            ("biHeight", LONG),
            ("biPlanes", WORD),
            ("biBitCount", WORD),
            ("biCompression", DWORD),
            ("biSizeImage", DWORD),
            ("biXPelsPerMeter", LONG),
            ("biYPelsPerMeter", LONG),
            ("biClrUsed", DWORD),
            ("biClrImportant", DWORD),
        ]

    class BITMAPINFO(ctypes.Structure):
        _fields_ = [
            ("bmiHeader", BITMAPINFOHEADER),
            ("bmiColors", DWORD * 3),
        ]


class ScreenCapture:
    """Captura retângulos da tela via Win32 GDI BitBlt com latência inferior a 2 ms."""

    def __init__(self) -> None:
        self._user32 = ctypes.windll.user32 if IS_WINDOWS else None
        self._gdi32 = ctypes.windll.gdi32 if IS_WINDOWS else None

    def capture_roi(self, center_x: int, center_y: int, width: int, height: int,
                    screen_size: tuple[int, int] | None = None) -> np.ndarray:
        """Captura um recorte de (width, height) centrado em (center_x, center_y).

        Retorna array BGR/RGB uint8 de shape (height, width, 3).
        """
        sw, sh = screen_size or (1920, 1080)
        x0 = int(center_x - width // 2)
        y0 = int(center_y - height // 2)

        # Clampeamento para manter dentro dos limites da tela
        src_x = max(0, min(sw - width, x0))
        src_y = max(0, min(sh - height, y0))

        if IS_WINDOWS and self._user32 and self._gdi32:
            try:
                return self._capture_gdi(src_x, src_y, width, height)
            except Exception as exc:
                log.debug("Fallback na captura GDI: %s", exc)

        # Fallback portátil via PIL ImageGrab caso disponível
        try:
            from PIL import ImageGrab
            bbox = (src_x, src_y, src_x + width, src_y + height)
            img = ImageGrab.grab(bbox=bbox)
            return np.array(img.convert("RGB"))
        except Exception:
            # Fallback seguro com imagem neutra em cinza
            return np.full((height, width, 3), 128, dtype=np.uint8)

    def _capture_gdi(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        u, g = self._user32, self._gdi32
        hdc_screen = u.GetDC(0)
        hdc_mem = g.CreateCompatibleDC(hdc_screen)
        hbm = g.CreateCompatibleBitmap(hdc_screen, w, h)
        old_hbm = g.SelectObject(hdc_mem, hbm)

        try:
            g.BitBlt(hdc_mem, 0, 0, w, h, hdc_screen, x, y, SRCCOPY | CAPTUREBLT)

            bmi = BITMAPINFO()
            bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER)
            bmi.bmiHeader.biWidth = w
            bmi.bmiHeader.biHeight = -h  # top-down DIB
            bmi.bmiHeader.biPlanes = 1
            bmi.bmiHeader.biBitCount = 32
            bmi.bmiHeader.biCompression = 0  # BI_RGB

            buffer = np.empty((h, w, 4), dtype=np.uint8)
            g.GetDIBits(
                hdc_mem,
                hbm,
                0,
                h,
                buffer.ctypes.data_as(c_void_p),
                byref(bmi),
                DIB_RGB_COLORS,
            )
            # Converte BGRA para RGB
            return np.ascontiguousarray(buffer[:, :, [2, 1, 0]])
        finally:
            g.SelectObject(hdc_mem, old_hbm)
            g.DeleteObject(hbm)
            g.DeleteDC(hdc_mem)
            u.ReleaseDC(0, hdc_screen)
