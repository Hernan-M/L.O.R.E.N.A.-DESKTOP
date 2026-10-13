"""Janela principal e orquestração dos fluxos Play / Calibrar."""
from __future__ import annotations

import logging
import queue
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import PhotoImage, messagebox
from tkinter.ttk import Button, Combobox, Frame, Label
from typing import Callable

from lorena import paths
from lorena.config import Settings
from lorena.domain.models import FrameResult
from lorena.infra.calibration_repo import CalibrationProfile, CalibrationRepository
from lorena.infra.platform import screen_pixel_pitch_mm
from lorena.mapping.gaze_mapper import GazeMapper
from lorena.services.calibration import CalibrationSession, Phase, calibration_targets, grid_targets
from lorena.services.cursor import Cursor
from lorena.services.cursor_session import CursorSession
from lorena.services.tracking_pipeline import TrackingPipeline
from lorena.ui import theme
from lorena.ui.calibration_view import CalibrationView
from lorena.ui.camera_preview import CameraPreview
from lorena.ui.magnifier_lens import MagnifierLens
from lorena.ui.status_overlay import StatusOverlay
from lorena.vision.camera import list_available_cameras

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
        self._lens = MagnifierLens(root, services.screen, services.settings.magnifier)
        self._cameras: list[tuple[int, str]] = []
        self._cb_source: Combobox | None = None
        self._preview: CameraPreview | None = None
        self._build()
        self._load_profile()
        self._root.after(POLL_MS, self._poll)

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        r, (sw, sh) = self._root, self._svc.screen
        r.title("L.O.R.E.N.A. — Assistente por Rastreio Ocular")
        win_w = max(720, min(860, sw // 2))
        win_h = max(530, min(620, sh // 2))
        r.geometry(f"{win_w}x{win_h}+{(sw - win_w) // 2}+{(sh - win_h) // 2}")
        r.minsize(660, 480)
        r.configure(bg=theme.BG)
        theme.apply()
        try:
            self._logo = PhotoImage(master=r, file=str(paths.LOGO_IMAGE))
            r.iconphoto(False, self._logo)
        except Exception as exc:
            log.debug("Não foi possível carregar o ícone da janela: %s", exc)
            self._logo = None

        container = Frame(r, style="TFrame")
        container.pack(expand=True, fill="both", padx=24, pady=16)

        # Cabeçalho com Logo e Título
        header = Frame(container, style="TFrame")
        header.pack(pady=(0, 14))

        if self._logo is not None:
            try:
                small_logo = self._logo.subsample(6, 6)
                logo = Label(header, image=small_logo, style="TLabel")
                logo.image = small_logo
                logo.pack(side="left", padx=(0, 14))
            except Exception as exc:
                log.debug("Não foi possível renderizar imagem do cabeçalho: %s", exc)


        header_text = Frame(header, style="TFrame")
        header_text.pack(side="left")
        Label(header_text, text="L.O.R.E.N.A.", style="Header.TLabel").pack(anchor="w")
        Label(
            header_text,
            text="Assistente de Acessibilidade por Rastreio Ocular",
            style="Subtitle.TLabel",
        ).pack(anchor="w")

        # Card Central: Configuração + Preview
        card = Frame(container, style="Card.TFrame")
        card.pack(fill="x", padx=10, pady=(0, 16))

        card_content = Frame(card, style="Card.TFrame")
        card_content.pack(padx=24, pady=16, fill="x")

        # Coluna Esquerda: Dispositivo e Status da Calibração
        left_col = Frame(card_content, style="Card.TFrame")
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 24))

        Label(left_col, text="DISPOSITIVO DE CAPTURA", style="CardHeader.TLabel").pack(anchor="w", pady=(0, 8))

        source_row = Frame(left_col, style="Card.TFrame")
        source_row.pack(anchor="w", pady=(0, 14))

        Label(source_row, text="Câmera:", style="Card.TLabel").pack(side="left", padx=(0, 8))

        self._cameras = list_available_cameras()
        log.info("Verificação de fontes ao iniciar: %d fontes detectadas: %s", len(self._cameras), self._cameras)

        current_idx = self._svc.settings.camera.index
        current_is_color = any(idx == current_idx and "RGB" in label for idx, label in self._cameras)
        has_color_available = bool(self._cameras and "RGB" in self._cameras[0][1])

        if not any(idx == current_idx for idx, _ in self._cameras) and self._cameras:
            current_idx = self._cameras[0][0]
            log.warning("Câmera configurada %d indisponível; alternando para fonte %d (%s)",
                        self._svc.settings.camera.index, current_idx, self._cameras[0][1])
            self._svc.settings.camera.index = current_idx
            try:
                self._svc.pipeline.set_camera_index(current_idx)
                self._svc.settings.save(paths.CONFIG_FILE)
                log.info("Configuração atualizada com fonte padrão válida: índice %d", current_idx)
            except Exception as exc:
                log.warning("Falha ao salvar configuração: %s", exc)
        elif not current_is_color and has_color_available:
            current_idx = self._cameras[0][0]
            log.info("Câmera padrão %d é IR/baixa luz; alternando automaticamente para câmera RGB (índice %d: %s)",
                     self._svc.settings.camera.index, current_idx, self._cameras[0][1])
            self._svc.settings.camera.index = current_idx
            try:
                self._svc.pipeline.set_camera_index(current_idx)
                self._svc.settings.save(paths.CONFIG_FILE)
            except Exception as exc:
                log.warning("Falha ao salvar configuração: %s", exc)
        else:
            log.info("Fonte de câmera confirmada para execução: índice %d", current_idx)

        cb_values = [label for _, label in self._cameras] if self._cameras else ["Nenhuma câmera"]
        self._cb_source = Combobox(
            source_row,
            values=cb_values,
            state="readonly" if self._cameras else "disabled",
            width=28,
        )
        selected_idx = 0
        for i, (idx, _) in enumerate(self._cameras):
            if idx == current_idx:
                selected_idx = i
                break
        if self._cameras:
            self._cb_source.current(selected_idx)
        self._cb_source.bind("<<ComboboxSelected>>", self._on_source_changed)
        self._cb_source.pack(side="left")

        self._btn_refresh = Button(
            source_row,
            text="🔄",
            width=3,
            command=self._on_refresh_cameras,
            cursor="hand2",
        )
        self._btn_refresh.pack(side="left", padx=(6, 0))

        Label(left_col, text="STATUS DO SISTEMA", style="CardHeader.TLabel").pack(anchor="w", pady=(6, 4))
        self._status = Label(left_col, style="StatusWarn.TLabel")
        self._status.pack(anchor="w")

        # Coluna Direita: Pré-visualização da Câmera
        right_col = Frame(card_content, style="Card.TFrame")
        right_col.pack(side="right")

        Label(right_col, text="PRÉ-VISUALIZAÇÃO", style="CardHeader.TLabel").pack(anchor="center", pady=(0, 6))
        self._preview = CameraPreview(right_col, camera_index=current_idx, width=180, height=135, style="Card.TFrame")
        self._preview.pack(anchor="center")

        # Painel de Ações
        buttons = Frame(container, style="TFrame")
        buttons.pack(pady=(4, 12))

        # Se não há câmeras conectadas, botões de ação iniciam desativados
        btn_state = "normal" if self._cameras else "disabled"
        self._btn_play = Button(
            buttons,
            text="▶  Iniciar Rastreio",
            style="TButton",
            command=self._on_play,
            cursor="hand2",
            state=btn_state,
        )
        self._btn_play.grid(row=0, column=0, padx=8, pady=4)

        self._btn_cal = Button(
            buttons,
            text="🎯  Calibrar Olhar",
            style="Secondary.TButton",
            command=self._on_calibrate,
            cursor="hand2",
            state=btn_state,
        )
        self._btn_cal.grid(row=0, column=1, padx=8, pady=4)

        # Rodapé informativo
        footer = Frame(container, style="TFrame")
        footer.pack()
        Label(
            footer,
            text="Dica: Pressione ESC para sair do rastreio • Pisque o olho esquerdo para clicar",
            style="Hint.TLabel",
        ).pack()

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
            self._status.configure(
                text="⚠  Nenhuma calibração ativa — calibre antes de iniciar.",
                style="StatusWarn.TLabel",
            )
            return
        self._svc.pipeline.extractor.openness.set_baseline(profile.open_ear_left, profile.open_ear_right)
        when = time.strftime("%d/%m/%Y %H:%M", time.localtime(profile.created_at))
        err = self._report_error(profile)
        self._status.configure(
            text=f"✓  Calibrado em {when} • Erro médio ≈ {err:.0f} px",
            style="StatusOk.TLabel",
        )

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
        if self._cb_source is not None:
            self._cb_source.configure(state="disabled" if busy else "readonly")

    def _on_source_changed(self, _event=None) -> None:
        if self._cb_source is None or not self._cameras:
            return
        sel_idx = self._cb_source.current()
        if sel_idx < 0 or sel_idx >= len(self._cameras):
            return
        cam_idx, cam_label = self._cameras[sel_idx]
        if cam_idx == self._svc.settings.camera.index and self._preview is not None and self._preview.is_running:
            return
        log.info("Fonte de captura selecionada pelo usuário: %s (índice %d)", cam_label, cam_idx)
        self._svc.settings.camera.index = cam_idx
        try:
            self._svc.pipeline.set_camera_index(cam_idx)
        except RuntimeError as exc:
            messagebox.showwarning("Aviso", str(exc))
            return
        if self._preview is not None:
            self._preview.set_camera_index(cam_idx)
        try:
            self._svc.settings.save(paths.CONFIG_FILE)
            log.info("Configuração salva com nova câmera em %s", paths.CONFIG_FILE)
        except Exception as exc:
            log.warning("Falha ao salvar configuração: %s", exc)

    def _on_refresh_cameras(self) -> None:
        """Atualiza sob demanda a lista de fontes de captura detectadas (hotplug)."""
        log.info("Atualização manual de fontes de captura solicitada pelo usuário.")
        self._cameras = list_available_cameras()
        cb_values = [label for _, label in self._cameras] if self._cameras else ["Nenhuma câmera"]
        if self._cb_source is not None:
            self._cb_source["values"] = cb_values
            self._cb_source.configure(state="readonly" if self._cameras else "disabled")

            current_idx = self._svc.settings.camera.index
            found = False
            for i, (idx, _) in enumerate(self._cameras):
                if idx == current_idx:
                    self._cb_source.current(i)
                    found = True
                    break
            if not found and self._cameras:
                self._cb_source.current(0)
                new_idx = self._cameras[0][0]
                self._svc.settings.camera.index = new_idx
                try:
                    self._svc.pipeline.set_camera_index(new_idx)
                except Exception:
                    pass
                if self._preview is not None:
                    self._preview.set_camera_index(new_idx)
                try:
                    self._svc.settings.save(paths.CONFIG_FILE)
                except Exception:
                    pass

        if hasattr(self, "_btn_play") and self._btn_play is not None:
            btn_state = "normal" if self._cameras else "disabled"
            self._btn_play.configure(state=btn_state)
            self._btn_cal.configure(state=btn_state)

    # ------------------------------------------------------------------- play
    def _on_play(self) -> None:
        if self._svc.pipeline.running:
            return
        if not self._cameras:
            messagebox.showwarning("Aviso", "Nenhuma câmera conectada para iniciar o rastreio.")
            return
        if self._profile is None:
            messagebox.showwarning("Aviso", "Calibre o sistema ao menos uma vez antes de iniciar o rastreio.")
            return
        pitch_mm = screen_pixel_pitch_mm()
        session = CursorSession(
            self._profile.mapper,
            self._svc.cursor,
            self._svc.settings,
            calibration_luma=self._profile.calibration_luma,
            pixel_pitch_mm=pitch_mm,
        )
        # Encerra o preview; Camera.open() já realiza retries com backoff em segundo plano
        if self._preview is not None:
            self._preview.stop()
        try:
            self._svc.pipeline.start(session, stop_on_escape=True)
        except RuntimeError as exc:
            if self._preview is not None:
                self._preview.start()
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
            if self._lens is not None and result.status is not None:
                self._lens.update(result.status.magnifier)

    def _stop_tracking(self) -> None:
        self._listener = None
        self._svc.pipeline.stop()
        self._overlay.hide()
        if self._lens is not None:
            self._lens.hide()
        self._set_busy(False)
        self._root.deiconify()
        if self._preview is not None:
            self._preview.start()

    # -------------------------------------------------------------- calibrate
    def _on_calibrate(self) -> None:
        if self._svc.pipeline.running:
            return
        if not self._cameras:
            messagebox.showwarning("Aviso", "Nenhuma câmera conectada para calibrar o sistema.")
            return
        cfg = self._svc.settings.calibration
        targets, pose_start, hints = calibration_targets(self._svc.screen, cfg)
        session = CalibrationSession(targets, cfg, pose_start=pose_start, pose_hints=hints)
        view = CalibrationView(self._root, session, self._svc.screen, self._on_calibration_finished)

        def listener(result: FrameResult) -> None:
            if result.error:
                view.close(cancelled=True)
                messagebox.showerror("Erro", result.error)
            else:
                view.on_result(result)

        # Encerra o preview; Camera.open() roda na thread de visão
        if self._preview is not None:
            self._preview.stop()
        try:
            self._svc.pipeline.start(None)
        except RuntimeError as exc:
            if self._preview is not None:
                self._preview.start()
            view.close(cancelled=True)
            messagebox.showerror("Erro", str(exc))
            return
        self._listener = listener
        self._set_busy(True)

    def _on_calibration_finished(self, session: CalibrationSession | None) -> None:
        self._listener = None
        self._svc.pipeline.stop()
        self._set_busy(False)
        if self._preview is not None:
            self._preview.start()
        if session is None:
            return
        if session.phase is Phase.FAILED:
            messagebox.showerror("Erro", "Não foi possível calibrar: muitos pontos sem leitura estável.\n"
                                         "Verifique a iluminação e a distância da câmera.")
            return
        cfg = self._svc.settings.calibration
        mapper = GazeMapper(
            degree=cfg.polynomial_degree,
            ridge_lambda=cfg.ridge_lambda,
            use_head_pose=cfg.use_head_pose,
            screen_size=self._svc.screen,
            use_lid=getattr(cfg, "use_lid", False),
            auto_select=getattr(cfg, "auto_select", False),
        )
        try:
            report = mapper.fit(session.samples, pose_ref=session.pose_ref)
        except (ValueError, ArithmeticError) as exc:
            log.exception("Falha no ajuste da calibração")
            messagebox.showerror("Erro", f"Falha ao calcular a calibração: {exc}")
            return
        open_left, open_right = session.open_ear_baseline() or (None, None)
        luma = session.face_luma_baseline()
        report.face_luma = luma
        profile = CalibrationProfile(
            mapper=mapper,
            open_ear_left=open_left or self._svc.settings.blink.default_open_ear,
            open_ear_right=open_right or self._svc.settings.blink.default_open_ear,
            screen_size=self._svc.screen,
            report=report,
            calibration_luma=luma,
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
        if self._preview is not None:
            self._preview.stop()
        self._svc.pipeline.stop()
        self._root.destroy()
