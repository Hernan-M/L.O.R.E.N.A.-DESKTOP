"""Tela de calibração em tela cheia (Tk, thread principal).

Só desenha o estado da ``CalibrationSession``; não contém lógica de cálculo.
O usuário apenas olha para o ponto: não há clique nem ENTER (que desviavam o
olhar e eram contraditórios para quem não usa o mouse).
"""
from __future__ import annotations

import tkinter as tk
from typing import Callable

from lorena.domain.models import FrameResult
from lorena.services.calibration import CalibrationSession, Phase
from lorena.ui import theme


class CalibrationView:
    def __init__(self, root: tk.Tk, session: CalibrationSession, screen: tuple[int, int],
                 on_finish: Callable[[CalibrationSession | None], None]) -> None:
        self._session = session
        self._on_finish = on_finish
        self._screen = screen
        self._closed = False

        self._win = tk.Toplevel(root)
        self._win.attributes("-fullscreen", True)
        self._win.attributes("-topmost", True)
        self._win.configure(bg="white", cursor="none")
        self._canvas = tk.Canvas(self._win, bg="white", highlightthickness=0)
        self._canvas.pack(fill="both", expand=True)
        self._win.bind("<Escape>", lambda _e: self.close(cancelled=True))
        self._win.protocol("WM_DELETE_WINDOW", lambda: self.close(cancelled=True))
        self._win.focus_force()
        self._draw()

    def on_result(self, result: FrameResult) -> None:
        if self._closed:
            return
        self._session.update(result.features, result.timestamp)
        self._draw()
        if self._session.finished:
            self.close(cancelled=False)

    def close(self, cancelled: bool) -> None:
        if self._closed:
            return
        self._closed = True
        self._win.destroy()
        self._on_finish(None if cancelled else self._session)

    def _draw(self) -> None:
        c, v, s = self._canvas, self._session.view(), self._session
        w, h = self._screen
        c.delete("all")
        for i, (x, y) in enumerate(s.targets):
            color = "#c9d6df" if i > v.index else ("#eb5757" if i in s.skipped else "#3ccf7a")
            if i != v.index:
                c.create_oval(x - 8, y - 8, x + 8, y + 8, fill=color, width=0)

        if v.pose_hint:
            title = f"Passe 2: Vire levemente a cabeça para {v.pose_hint} e olhe para o alvo"
        else:
            title = "Olhe fixamente para o ponto azul até ele ficar verde. ESC cancela."
        c.create_text(w // 2, int(h * 0.22), text=title, font=(theme.FONT, 20), fill="#1b2a38")

        if v.phase is Phase.WAIT_FACE or not v.face_ok:
            status, color = "Rosto não identificado — alinhe-se com a câmera e mantenha os olhos abertos", theme.ERROR
        elif v.pose_hint:
            if v.pose_dev_deg < v.pose_min_deg:
                status = f"Ponto {v.index + 1} de {v.total} • Giro da cabeça: {v.pose_dev_deg:.1f}° (mínimo {v.pose_min_deg:.1f}° — vire um pouco mais)"
                color = "#d97706"
            else:
                status = f"Ponto {v.index + 1} de {v.total} • Giro: {v.pose_dev_deg:.1f}° ✓ — mantenha e fixe o olhar"
                color = "#00a37a"
        else:
            status, color = f"Ponto {v.index + 1} de {v.total}" + (f"  (repetindo {v.retry})" if v.retry else ""), "#1b6d8c"
        c.create_text(w // 2, int(h * 0.22) + 44, text=status, font=(theme.FONT, 16), fill=color)

        if v.target is None:
            return
        x, y = v.target
        # Círculo que encolhe atrai e mantém o olhar no centro do alvo.
        radius = 34 - 22 * v.progress
        fill = "#1e88e5" if v.phase is not Phase.COLLECT else "#00a37a"
        c.create_oval(x - 40, y - 40, x + 40, y + 40, outline="#d0dbe4", width=3)
        c.create_arc(x - 40, y - 40, x + 40, y + 40, start=90, extent=-359.9 * v.progress,
                     style="arc", outline=fill, width=5)
        c.create_oval(x - radius, y - radius, x + radius, y + radius, fill=fill, width=0)
        c.create_oval(x - 3, y - 3, x + 3, y + 3, fill="white", width=0)
