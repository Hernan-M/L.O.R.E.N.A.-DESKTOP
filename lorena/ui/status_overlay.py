"""Overlay pequeno, sempre no topo, com o estado do rastreio (não bloqueante).

Substitui os MessageBox bloqueantes e a janela OpenCV em tela cheia que era
recriada a cada clique.
"""
from __future__ import annotations

import tkinter as tk

from lorena.domain.models import GestureEvent, SessionStatus, TrackingState
from lorena.ui import theme

_STATE_TEXT = {
    TrackingState.TRACKING: ("Rastreando", theme.OK),
    TrackingState.DEGRADED: ("Rastreio instável — melhore a luz", theme.WARN),
    TrackingState.LOST: ("Rosto não encontrado — aproxime-se", theme.ERROR),
    TrackingState.PAUSED: ("Pausado (feche os 2 olhos para retomar)", theme.WARN),
    TrackingState.IDLE: ("Parado", theme.FG),
}
_EVENT_TEXT = {
    GestureEvent.CLICK: "Clique",
    GestureEvent.RIGHT_CLICK: "Clique direito",
    GestureEvent.DWELL_CLICK: "Clique (fixação)",
}


class StatusOverlay:
    WIDTH, HEIGHT = 340, 74

    def __init__(self, root: tk.Tk, screen: tuple[int, int]) -> None:
        self._win = tk.Toplevel(root)
        self._win.withdraw()
        self._win.overrideredirect(True)
        self._win.attributes("-topmost", True)
        self._win.attributes("-alpha", 0.88)
        self._win.configure(bg=theme.BG)
        self._win.geometry(f"{self.WIDTH}x{self.HEIGHT}+{screen[0] - self.WIDTH - 16}+16")
        self._state = tk.Label(self._win, font=(theme.FONT, 11, "bold"), bg=theme.BG, fg=theme.FG, anchor="w")
        self._state.pack(fill="x", padx=10, pady=(6, 0))
        self._info = tk.Label(self._win, font=(theme.FONT, 9), bg=theme.BG, fg="#9fb3c8", anchor="w")
        self._info.pack(fill="x", padx=10)
        self._bar = tk.Canvas(self._win, height=6, bg="#17344d", highlightthickness=0)
        self._bar.pack(fill="x", padx=10, pady=(4, 6))
        self._event_until = 0.0
        self._event_text = ""

    def show(self) -> None:
        self._win.deiconify()
        self._win.lift()

    def hide(self) -> None:
        self._win.withdraw()

    def update(self, status: SessionStatus | None, fps: float, now: float) -> None:
        if status is None:
            return
        text, color = _STATE_TEXT.get(status.state, ("", theme.FG))
        if status.last_event in _EVENT_TEXT:
            self._event_text, self._event_until = _EVENT_TEXT[status.last_event], now + 0.8
        if now < self._event_until:
            text, color = self._event_text, theme.ACCENT
        self._state.configure(text=text, fg=color)
        self._info.configure(text=f"{fps:4.1f} FPS  •  ESC para sair")
        self._bar.delete("all")
        if status.dwell_progress > 0:
            width = self._bar.winfo_width() * status.dwell_progress
            self._bar.create_rectangle(0, 0, width, 6, fill=theme.ACCENT, width=0)
