"""Janela principal e orquestração dos fluxos Play / Calibrar."""
from __future__ import annotations

import logging
import queue
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import PhotoImage, messagebox
from tkinter.ttk import Button, Frame, Label
from typing import Callable

from lorena import paths
from lorena.config import Settings
from lorena.domain.models import FrameResult
from lorena.infra.calibration_repo import CalibrationProfile, CalibrationRepository
from lorena.mapping.gaze_mapper import GazeMapper
from lorena.services.calibration import CalibrationSession, Phase, grid_targets
from lorena.services.cursor import Cursor
from lorena.services.cursor_session import CursorSession
from lorena.services.tracking_pipeline import TrackingPipeline
from lorena.ui import theme
from lorena.ui.calibration_view import CalibrationView
from lorena.ui.status_overlay import StatusOverlay

log = logging.getLogger(__name__)

POLL_MS = 15
WARN_ERROR_PX = 120


@dataclass
class AppServices:
    settings: Settings
    screen: tuple[int, int]
    pipeline: TrackingPipeline
    repository: CalibrationRepository
    cursor: Cursor


class MainWindow:
    def __init__(self, root: tk.Tk, services: AppServices) -> None:
        self._root = root
        self._svc = services
        self._profile: CalibrationProfile | None = None
        self._listener: Callable[[FrameResult], None] | None = None
        self._overlay = StatusOverlay(root, services.screen)
        self._build()
        self._load_profile()
        self._root.after(POLL_MS, self._poll)

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        r, (sw, sh) = self._root, self._svc.screen
        r.title("L.O.R.E.N.A.")
        r.geometry(f"{sw // 2}x{sh // 2}+{sw // 4}+{sh // 4}")
        r.configure(bg=theme.BG)
        theme.apply()
        self._logo = PhotoImage(file=str(paths.LOGO_IMAGE))
        r.iconphoto(False, self._logo)

        body = Frame(r)
        body.pack(expand=True)
        small_logo = self._logo.subsample(5, 5)
        logo = Label(body, image=small_logo)
        logo.image = small_logo
        logo.pack(pady=(0, 16))
        Label(body, text="Bem-vindo(a) ao LORENA,").pack()
        Label(body, text="seu assistente virtual por eyeTracking").pack()
        Label(body, text="Escolha uma opção:").pack(pady=10)

        buttons = Frame(body)
        buttons.pack()
        self._btn_play = Button(buttons, text="Play", command=self._on_play, cursor="hand2")
        self._btn_play.grid(row=0, column=0, padx=10, pady=10)
        self._btn_cal = Button(buttons, text="Calibrar", command=self._on_calibrate, cursor="hand2")
        self._btn_cal.grid(row=0, column=1, padx=10, pady=10)
        self._status = Label(body, style="Small.TLabel")
        self._status.pack(pady=(8, 0))

        r.bind("<Return>", lambda _e: getattr(r.focus_get(), "invoke", lambda: None)())
        r.protocol("WM_DELETE_WINDOW", self._on_close)
        self._btn_play.focus_set()

    def _load_profile(self) -> None:
        profile = self._svc.repository.load(self._svc.settings.profile)
        if profile is not None and tuple(profile.screen_size) != tuple(self._svc.screen):
            log.info("Resolução mudou desde a calibração; é preciso recalibrar.")
            profile = None
        self._set_profile(profile)

    def _set_profile(self, profile: CalibrationProfile | None) -> None:
        self._profile = profile
        if profile is None:
            self._status.configure(text="Nenhuma calibração válida — calibre antes de iniciar.")
            return
        self._svc.pipeline.extractor.openness.set_baseline(profile.open_ear_left, profile.open_ear_right)
        when = time.strftime("%d/%m/%Y %H:%M", time.localtime(profile.created_at))
        self._status.configure(text=f"Calibrado em {when} • erro médio ≈ {self._report_error(profile):.0f} px")

    @staticmethod
    def _report_error(profile: CalibrationProfile) -> float:
        rep = profile.report
        return rep.loo_mean_error_px if rep.loo_mean_error_px is not None else rep.mean_error_px

    # ------------------------------------------------------------------- loop
    def _poll(self) -> None:
        q = self._svc.pipeline.results
        for _ in range(64):
            try:
                result = q.get_nowait()
            except queue.Empty:
                break
            if self._listener is not None:
                self._listener(result)
        self._root.after(POLL_MS, self._poll)

    def _set_busy(self, busy: bool) -> None:
        state = ["disabled"] if busy else ["!disabled"]
        self._btn_play.state(state)
        self._btn_cal.state(state)

    # ------------------------------------------------------------------- play
    def _on_play(self) -> None:
        if self._svc.pipeline.running:
            return
        if self._profile is None:
            messagebox.showwarning("Aviso", "Calibre o sistema ao menos uma vez antes de iniciar o rastreio.")
            return
        session = CursorSession(self._profile.mapper, self._svc.cursor, self._svc.settings)
        messagebox.showinfo("Aviso", "Para sair do rastreio ocular aperte ESC.\n"
                                     "Clique: pisque só o olho esquerdo.\n"
                                     "Pausar/retomar: feche os dois olhos por ~1,5 s.")
        try:
            self._svc.pipeline.start(session, stop_on_escape=True)
        except RuntimeError as exc:
            messagebox.showerror("Erro", str(exc))
            return
        self._listener = self._on_tracking_result
        self._set_busy(True)
        self._overlay.show()
        self._root.iconify()

    def _on_tracking_result(self, result: FrameResult) -> None:
        if result.error:
            self._stop_tracking()
            messagebox.showerror("Erro de rastreio", result.error)
        elif result.stopped_by_user:
            self._stop_tracking()
        else:
            self._overlay.update(result.status, result.fps, time.monotonic())

    def _stop_tracking(self) -> None:
        self._listener = None
        self._svc.pipeline.stop()
        self._overlay.hide()
        self._set_busy(False)
        self._root.deiconify()

    # -------------------------------------------------------------- calibrate
    def _on_calibrate(self) -> None:
        if self._svc.pipeline.running:
            return
        cfg = self._svc.settings.calibration
        session = CalibrationSession(grid_targets(self._svc.screen, cfg.grid, cfg.margin_px), cfg)
        view = CalibrationView(self._root, session, self._svc.screen, self._on_calibration_finished)

        def listener(result: FrameResult) -> None:
            if result.error:
                view.close(cancelled=True)
                messagebox.showerror("Erro", result.error)
            else:
                view.on_result(result)

        try:
            self._svc.pipeline.start(None)
        except RuntimeError as exc:
            view.close(cancelled=True)
            messagebox.showerror("Erro", str(exc))
            return
        self._listener = listener
        self._set_busy(True)

    def _on_calibration_finished(self, session: CalibrationSession | None) -> None:
        self._listener = None
        self._svc.pipeline.stop()
        self._set_busy(False)
        if session is None:
            return
        if session.phase is Phase.FAILED:
            messagebox.showerror("Erro", "Não foi possível calibrar: muitos pontos sem leitura estável.\n"
                                         "Verifique a iluminação e a distância da câmera.")
            return
        cfg = self._svc.settings.calibration
        mapper = GazeMapper(cfg.polynomial_degree, cfg.ridge_lambda, cfg.use_head_pose, self._svc.screen)
        try:
            report = mapper.fit(session.samples)
        except (ValueError, ArithmeticError) as exc:
            log.exception("Falha no ajuste da calibração")
            messagebox.showerror("Erro", f"Falha ao calcular a calibração: {exc}")
            return
        open_left, open_right = session.open_ear_baseline() or (None, None)
        profile = CalibrationProfile(
            mapper=mapper,
            open_ear_left=open_left or self._svc.settings.blink.default_open_ear,
            open_ear_right=open_right or self._svc.settings.blink.default_open_ear,
            screen_size=self._svc.screen,
            report=report,
        )
        self._svc.repository.save(self._svc.settings.profile, profile)
        self._set_profile(profile)
        log.info("Calibração salva: %s", report)

        error = self._report_error(profile)
        msg = f"Calibração salva!\nErro médio estimado: {error:.0f} px ({report.num_points} pontos)."
        if session.skipped:
            msg += f"\nPontos ignorados: {', '.join(str(i + 1) for i in session.skipped)}."
        if error > WARN_ERROR_PX:
            messagebox.showwarning("Calibração imprecisa", msg + "\n\nRecomenda-se recalibrar com melhor iluminação.")
        else:
            messagebox.showinfo("Calibração", msg)

    # ------------------------------------------------------------------ close
    def _on_close(self) -> None:
        self._listener = None
        self._svc.pipeline.stop()
        self._root.destroy()
