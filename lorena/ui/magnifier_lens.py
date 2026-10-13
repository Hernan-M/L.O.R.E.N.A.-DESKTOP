"""Lente de Aumento Inteligente (Assistive Zoom Lens) com animações Ease-In e Ease-Out.

Exibe um overlay circular não-bloqueante sobre a região de fixação com ampliação
(Zoom-and-Click), permitindo mira de alta precisão em micro-alvos (15.6" e botões pequenos).
"""
from __future__ import annotations

import logging
import math
import sys
import tkinter as tk
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from lorena.config import MagnifierSettings
from lorena.infra.screen_capture import ScreenCapture
from lorena.services.magnifier_controller import (
    MagnifierState,
    MagnifierStatus,
    ease_in_cubic,
    ease_out_cubic,
)
from lorena.ui import theme

log = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == "win32"
TRANSPARENT_COLOR = "#000001"


class MagnifierLens:
    """Overlay visual flutuante para a Lupa Inteligente (Zoom-and-Click)."""

    def __init__(
        self,
        root: tk.Tk,
        screen_size: tuple[int, int],
        settings: MagnifierSettings | None = None,
        screen_capture: ScreenCapture | None = None,
    ) -> None:
        self._root = root
        self._screen = screen_size
        self._s = settings or MagnifierSettings()
        self._capture = screen_capture or ScreenCapture()

        self._win = tk.Toplevel(root)
        self._win.withdraw()
        self._win.overrideredirect(True)
        self._win.attributes("-topmost", True)
        if IS_WINDOWS:
            try:
                self._win.attributes("-transparentcolor", TRANSPARENT_COLOR)
            except Exception as exc:
                log.debug("Falha ao definir transparentcolor: %s", exc)

        self._diameter = int(self._s.lens_radius_px * 2)
        self._win.geometry(f"{self._diameter}x{self._diameter}+0+0")

        self._canvas = tk.Canvas(
            self._win,
            width=self._diameter,
            height=self._diameter,
            bg=TRANSPARENT_COLOR,
            highlightthickness=0,
            bd=0,
        )
        self._canvas.pack(fill="both", expand=True)

        self._active_status: MagnifierStatus | None = None
        self._cached_roi_rgba: Image.Image | None = None
        self._current_photo: ImageTk.PhotoImage | None = None
        self._anim_job: str | None = None
        self._is_opening = False
        self._is_closing = False
        self._is_fully_open = False
        self._center: tuple[float, float] = (0.0, 0.0)
        self._win_x = 0
        self._win_y = 0

    @property
    def is_visible(self) -> bool:
        return self._is_fully_open or self._is_opening or self._is_closing

    def update(self, status: MagnifierStatus | None) -> None:
        """Recebe o estado atual da lupa vindo do pipeline de visão."""
        if not self._s.enabled or status is None or status.state is MagnifierState.IDLE:
            if self._is_fully_open or self._is_opening:
                self._start_ease_out()
            elif not self._is_closing:
                self.hide()
            self._active_status = None
            return

        self._active_status = status

        if status.state is MagnifierState.CHARGING:
            self._draw_charging_ring(status)
            return

        if status.state is MagnifierState.ACTIVE:
            if not self._is_fully_open and not self._is_opening and not self._is_closing:
                self._open_lens(status)
            elif self._is_fully_open:
                self._redraw_active(status)

    def _draw_charging_ring(self, status: MagnifierStatus) -> None:
        """Exibe o anel de pré-engate da lupa (1.0s a 1.5s de fixação contínua)."""
        if self._is_opening or self._is_fully_open or status.center is None:
            return

        cx, cy = status.center
        ring_r = 28
        ring_d = ring_r * 2
        wx = int(cx - ring_r)
        wy = int(cy - ring_r)
        sw, sh = self._screen
        wx = max(0, min(sw - ring_d, wx))
        wy = max(0, min(sh - ring_d, wy))

        self._win.geometry(f"{ring_d}x{ring_d}+{wx}+{wy}")
        self._win.deiconify()
        self._win.lift()

        self._canvas.delete("all")
        # Anel de fundo translúcido
        self._canvas.create_oval(
            3, 3, ring_d - 3, ring_d - 3, outline="#17344d", width=3
        )
        # Arco de progresso (0 a 359 graus)
        extent = max(1.0, min(359.0, status.charge_progress * 359.0))
        self._canvas.create_arc(
            3,
            3,
            ring_d - 3,
            ring_d - 3,
            start=90,
            extent=-extent,
            outline=theme.ACCENT,
            style="arc",
            width=3,
        )

    def _open_lens(self, status: MagnifierStatus) -> None:
        """Inicia a captura da tela e animação Ease-In."""
        if status.center is None:
            return

        self._cancel_animation()
        self._center = status.center
        cx, cy = self._center
        self._diameter = int(status.lens_radius_px * 2)

        # Clampeamento da janela para não sair da tela
        sw, sh = self._screen
        self._win_x = max(0, min(sw - self._diameter, int(cx - self._diameter // 2)))
        self._win_y = max(0, min(sh - self._diameter, int(cy - self._diameter // 2)))

        self._win.geometry(
            f"{self._diameter}x{self._diameter}+{self._win_x}+{self._win_y}"
        )
        self._win.deiconify()
        self._win.lift()

        # Captura ultrarrápida do ROI na tela real
        zoom = max(1.0, float(status.zoom_factor))
        roi_size = max(32, int(round(self._diameter / zoom)))
        roi_rgb = self._capture.capture_roi(
            int(cx), int(cy), roi_size, roi_size, self._screen
        )

        # Converte para PIL e cria a máscara circular
        roi_img = Image.fromarray(roi_rgb).resize(
            (self._diameter, self._diameter), Image.BILINEAR
        )
        mask = Image.new("L", (self._diameter, self._diameter), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, self._diameter, self._diameter), fill=255)

        roi_rgba = roi_img.convert("RGBA")
        roi_rgba.putalpha(mask)
        self._cached_roi_rgba = roi_rgba

        self._is_opening = True
        self._is_closing = False
        self._is_fully_open = False

        duration = max(50, self._s.ease_duration_ms)
        total_steps = max(4, duration // 16)
        self._step_ease_in(0, total_steps)

    def _step_ease_in(self, step: int, total_steps: int) -> None:
        if not self._is_opening or self._cached_roi_rgba is None:
            return

        t = (step + 1) / total_steps
        scale = ease_in_cubic(t)

        self._render_lens_frame(scale)

        if step + 1 < total_steps:
            self._anim_job = self._root.after(
                16, self._step_ease_in, step + 1, total_steps
            )
        else:
            self._is_opening = False
            self._is_fully_open = True
            self._anim_job = None
            if self._active_status is not None:
                self._redraw_active(self._active_status)

    def _start_ease_out(self) -> None:
        """Inicia animação suave Ease-Out de fechamento."""
        if self._is_closing or self._cached_roi_rgba is None:
            self.hide()
            return

        self._cancel_animation()
        self._is_opening = False
        self._is_fully_open = False
        self._is_closing = True

        duration = max(50, self._s.ease_duration_ms)
        total_steps = max(4, duration // 16)
        self._step_ease_out(0, total_steps)

    def _step_ease_out(self, step: int, total_steps: int) -> None:
        if not self._is_closing or self._cached_roi_rgba is None:
            self.hide()
            return

        t = (step + 1) / total_steps
        scale = ease_out_cubic(t)

        if scale <= 0.05 or step + 1 >= total_steps:
            self.hide()
            return

        self._render_lens_frame(scale)
        self._anim_job = self._root.after(
            16, self._step_ease_out, step + 1, total_steps
        )

    def _render_lens_frame(self, scale: float) -> None:
        """Renderiza um frame da lente escalada pelo fator de animação."""
        if self._cached_roi_rgba is None:
            return

        D = self._diameter
        scaled_d = max(10, int(round(D * scale)))
        scaled_img = self._cached_roi_rgba.resize(
            (scaled_d, scaled_d), Image.BILINEAR
        )

        offset = (D - scaled_d) // 2

        self._current_photo = ImageTk.PhotoImage(scaled_img)
        self._canvas.delete("all")
        self._canvas.create_image(
            offset, offset, anchor="nw", image=self._current_photo
        )

        # Borda circular translúcida e elegante
        self._canvas.create_oval(
            offset + 1,
            offset + 1,
            offset + scaled_d - 1,
            offset + scaled_d - 1,
            outline=theme.ACCENT,
            width=3,
        )

        # Retículo central discreto quando a lente estiver aberta quase totalmente
        if scale > 0.8:
            mid = D // 2
            self._canvas.create_line(
                mid - 6, mid, mid + 6, mid, fill="#ffffff", width=1
            )
            self._canvas.create_line(
                mid, mid - 6, mid, mid + 6, fill="#ffffff", width=1
            )

    def _redraw_active(self, status: MagnifierStatus) -> None:
        """Atualiza elementos interativos dentro da lente já aberta."""
        if not self._is_fully_open or self._cached_roi_rgba is None:
            return

        # Se houver alvo projetado ou olhar atual, podemos atualizar o retículo
        self._render_lens_frame(1.0)
        if status.real_target is not None and status.center is not None:
            # Ponto de mira dentro da lente
            cx, cy = status.center
            rx, ry = status.real_target
            # offset relativo à janela da lente
            zoom = status.zoom_factor
            lx = (rx - cx) * zoom + self._diameter // 2
            ly = (ry - cy) * zoom + self._diameter // 2
            r_dot = 4
            self._canvas.create_oval(
                lx - r_dot,
                ly - r_dot,
                lx + r_dot,
                ly + r_dot,
                outline="#ffffff",
                fill=theme.ACCENT,
                width=1,
            )

    def _cancel_animation(self) -> None:
        if self._anim_job is not None:
            try:
                self._root.after_cancel(self._anim_job)
            except Exception:
                pass
            self._anim_job = None

    def dismiss(self) -> None:
        """Fecha a lente suavemente com animação Ease-Out."""
        self._start_ease_out()

    def hide(self) -> None:
        """Oculta imediatamente o overlay e reseta o estado da animação."""
        self._cancel_animation()
        self._is_opening = False
        self._is_closing = False
        self._is_fully_open = False
        self._cached_roi_rgba = None
        self._current_photo = None
        self._active_status = None
        self._canvas.delete("all")
        self._win.withdraw()
