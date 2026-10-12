"""Controle do cursor do sistema.

Usa a API Win32 diretamente: o ``pyautogui`` antigo dormia 100 ms
(``PAUSE``) a cada ``moveTo``, limitando o loop a ~5 FPS.
"""
from __future__ import annotations

import ctypes
from typing import Protocol

import numpy as np

from lorena.infra.platform import IS_WINDOWS

_LEFTDOWN, _LEFTUP, _RIGHTDOWN, _RIGHTUP = 0x0002, 0x0004, 0x0008, 0x0010


class Cursor(Protocol):
    def move_to(self, p: np.ndarray) -> None: ...
    def click(self, button: str = "left") -> None: ...


class Win32Cursor:
    def __init__(self) -> None:
        if not IS_WINDOWS:
            raise RuntimeError("Win32Cursor requer Windows.")
        self._u = ctypes.windll.user32

    def move_to(self, p: np.ndarray) -> None:
        self._u.SetCursorPos(int(round(p[0])), int(round(p[1])))

    def click(self, button: str = "left") -> None:
        down, up = (_RIGHTDOWN, _RIGHTUP) if button == "right" else (_LEFTDOWN, _LEFTUP)
        self._u.mouse_event(down, 0, 0, 0, 0)
        self._u.mouse_event(up, 0, 0, 0, 0)


class RecordingCursor:
    """Cursor falso para testes e modo de depuração."""

    def __init__(self) -> None:
        self.positions: list[tuple[float, float]] = []
        self.clicks: list[tuple[str, tuple[float, float] | None]] = []

    def move_to(self, p: np.ndarray) -> None:
        self.positions.append((float(p[0]), float(p[1])))

    def click(self, button: str = "left") -> None:
        self.clicks.append((button, self.positions[-1] if self.positions else None))
