"""Sessão de controle do cursor: features → pixel filtrado → cursor + gestos.

Roda na thread de visão (baixa latência). Estados de rastreio:
TRACKING → DEGRADED (baixa confiança) → LOST (sem rosto > lost_after_s).
Em LOST o cursor fica parado (não "voa") e o filtro é reiniciado na volta.

v0.3.1: parâmetros espaciais em graus de ângulo visual (escalam com o tamanho físico da
tela), congelamento breve quando a cabeça gira rápido, OutlierGate que aceita saltos
explicados pela pose, One Euro adaptativo ao ruído e zona morta menor nas bordas.
"""
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import replace

import numpy as np

from lorena.config import Settings
from lorena.domain.models import GazeFeatures, GestureEvent, SessionStatus, TrackingState
from lorena.mapping.filters import (
    DwellClicker,
    FixationStabilizer,
    OneEuroFilter,
    OneEuroFilter2D,
    OutlierGate,
    PositionHistory,
)
from lorena.mapping.gaze_mapper import GazeMapper
from lorena.mapping.geometry import DEFAULT_PITCH_MM, deg_to_px
from lorena.services.cursor import Cursor
from lorena.services.magnifier_controller import MagnifierController, MagnifierState
from lorena.vision.blink_detector import BlinkDetector

log = logging.getLogger(__name__)


class CursorSession:
    def __init__(self, mapper: GazeMapper, cursor: Cursor, settings: Settings,
                 calibration_luma: float | None = None, pixel_pitch_mm: float | None = None) -> None:
        self._mapper = mapper
        self._cursor = cursor
        self._s = settings
        f = settings.filters
        pitch = pixel_pitch_mm or DEFAULT_PITCH_MM
        ppd = deg_to_px(1.0, pitch, f.viewing_distance_mm)  # pixels por grau de ângulo visual
        jump_px = f.jump_deg * ppd if f.jump_deg > 0 else f.jump_px
        self._fix_radius = f.fixation_radius_deg * ppd if f.fixation_radius_deg > 0 else f.fixation_radius_px
        self._noise_ref_px = max(f.noise_ref_deg * ppd, 1.0)
        self._gate = OutlierGate(jump_px, f.confirm_frames)
        self._euro = OneEuroFilter2D(f.min_cutoff, f.beta, f.d_cutoff)
        self._stabilizer = FixationStabilizer(self._fix_radius)
        self._dwell = DwellClicker(settings.tracking.dwell_s)
        self._blink = BlinkDetector(settings.blink)
        self._magnifier = MagnifierController(settings.magnifier, screen_size=mapper.screen_size) if getattr(settings, "magnifier", None) else None
        self._history = PositionHistory()
        self._frozen_at: np.ndarray | None = None
        self._last_seen: float | None = None
        self._state = TrackingState.TRACKING
        self._paused = False
        # --- pré-filtros adaptativos de pose e pálpebra (D5) ---
        self._euro_yaw = OneEuroFilter(min_cutoff=1.2, beta=0.15, d_cutoff=1.0)
        self._euro_pitch = OneEuroFilter(min_cutoff=1.2, beta=0.15, d_cutoff=1.0)
        self._euro_hx = OneEuroFilter(min_cutoff=1.0, beta=0.10, d_cutoff=1.0)
        self._euro_hy = OneEuroFilter(min_cutoff=1.0, beta=0.10, d_cutoff=1.0)
        self._euro_lid = OneEuroFilter(min_cutoff=1.5, beta=0.20, d_cutoff=1.0)
        # --- cabeça / ruído / luz ---
        self._prev: GazeFeatures | None = None
        self._hold_until = 0.0
        self._jitter: deque[float] = deque(maxlen=30)
        self._last_raw: np.ndarray | None = None
        self._cal_luma = calibration_luma
        self._luma_ema: float | None = None
        self._noise_ema: float | None = None
        self._drift_since: float | None = None
        # --- projeção cinemática (Opção A + Frenagem D4) ---
        self._vel_filtered: np.ndarray = np.zeros(2, dtype=float)
        self._last_euro_p: np.ndarray | None = None
        self._last_euro_t: float | None = None
        self._saccade_dir: np.ndarray | None = None
        self._last_dispatch: np.ndarray | None = None

    @property
    def paused(self) -> bool:
        return self._paused

    def _reset_filters(self) -> None:
        self._gate.reset()
        self._euro.reset()
        self._stabilizer.reset()
        self._euro_yaw.reset()
        self._euro_pitch.reset()
        self._euro_hx.reset()
        self._euro_hy.reset()
        self._euro_lid.reset()
        if self._magnifier is not None:
            self._magnifier.reset()
        self._history.clear()
        self._jitter.clear()
        self._last_raw = None
        self._noise_ema = None
        self._prev = None
        self._hold_until = 0.0
        self._vel_filtered = np.zeros(2, dtype=float)
        self._last_euro_p = None
        self._last_euro_t = None
        self._saccade_dir = None
        self._last_dispatch = None

    def on_frame(self, f: GazeFeatures | None, t: float) -> SessionStatus:
        mode = self._s.tracking.click_mode
        event: GestureEvent | None = None
        dwell_progress = 0.0

        # --- presença do rosto ---------------------------------------------
        if f is None:
            self._blink.reset()
            self._frozen_at = None
            if self._magnifier is not None:
                self._magnifier.reset()
            if self._last_seen is None or t - self._last_seen > self._s.tracking.lost_after_s:
                if self._state is not TrackingState.LOST:
                    self._reset_filters()
                self._state = TrackingState.LOST
            return SessionStatus(TrackingState.PAUSED if self._paused else self._state)
        self._last_seen = t

        # --- gestos de piscada (rodam mesmo pausado, p/ permitir retomar) --
        event = self._blink.update(f.openness_left, f.openness_right, t)
        if event is GestureEvent.TOGGLE_PAUSE:
            self._paused = not self._paused
            log.info("Rastreio %s por gesto.", "pausado" if self._paused else "retomado")
            self._reset_filters()
            return SessionStatus(TrackingState.PAUSED if self._paused else TrackingState.TRACKING, 0.0, event)
        if self._paused:
            return SessionStatus(TrackingState.PAUSED)
        if mode == "dwell" and event in (GestureEvent.CLICK, GestureEvent.RIGHT_CLICK):
            event = None  # no modo dwell, wink não clica

        # --- congelamento durante o gesto ----------------------------------
        if self._blink.freeze:
            if self._frozen_at is None:
                self._frozen_at = self._history.at(t - self._s.blink.freeze_rollback_s)
                if self._frozen_at is not None:
                    self._cursor.move_to(self._frozen_at)
            return SessionStatus(self._state, 0.0, None)

        if event is not None:
            # Se a lupa estiver ativa no momento do clique (Zoom-and-Click):
            if (self._magnifier is not None and self._magnifier.state is MagnifierState.ACTIVE
                    and getattr(self._s.magnifier, "mode", "zoom_and_click") == "zoom_and_click"):
                ref_pos = self._frozen_at if self._frozen_at is not None else (self._last_raw if self._last_raw is not None else np.array([0.0, 0.0]))
                target = self._magnifier.map_to_real_screen(ref_pos)
                self._cursor.move_to(target)
                self._cursor.click("right" if event is GestureEvent.RIGHT_CLICK else "left")
                self._magnifier.dismiss()
            else:
                if self._frozen_at is not None:
                    self._cursor.move_to(self._frozen_at)
                self._cursor.click("right" if event is GestureEvent.RIGHT_CLICK else "left")
        self._frozen_at = None

        # --- pré-filtragem adaptativa de pose e abertura de pálpebra (D5) ---
        filt_yaw = self._euro_yaw(f.yaw, t)
        filt_pitch = self._euro_pitch(f.pitch, t)
        filt_hx = self._euro_hx(f.head_x, t)
        filt_hy = self._euro_hy(f.head_y, t)
        filt_lid = self._euro_lid(f.lid, t)
        f_clean = replace(f, yaw=filt_yaw, pitch=filt_pitch, head_x=filt_hx, head_y=filt_hy, lid=filt_lid)

        # --- movimento -------------------------------------------------------
        if not (math.isfinite(f_clean.h) and math.isfinite(f_clean.v)):
            return SessionStatus(self._state, 0.0, event)
        was_lost = self._state is TrackingState.LOST
        out_of_range = self._mapper.pose_out_of_range(f_clean)
        degraded = f_clean.quality < self._s.tracking.degraded_quality or out_of_range
        self._state = TrackingState.DEGRADED if degraded else TrackingState.TRACKING
        if was_lost:
            self._reset_filters()
        hint = "head" if out_of_range else ("light" if self._light_hint(f_clean, t) else None)

        # Giro rápido da cabeça: os landmarks ficam instáveis → congela por um instante.
        flt = self._s.filters
        if self._head_rate_deg_s(f_clean) > flt.head_rate_deg_s:
            self._hold_until = t + flt.head_hold_s
        prev, self._prev = self._prev, f_clean
        if t < self._hold_until:
            shift = self._mapper.pose_shift(prev, f_clean) if prev is not None else None
            if shift is not None:
                self._gate.apply_shift(shift)
            self._vel_filtered = np.zeros(2, dtype=float)
            self._last_euro_p = None
            self._last_euro_t = None
            return SessionStatus(self._state, 0.0, event, hint)

        p_raw = self._mapper.predict(f_clean)
        self._update_noise(p_raw)
        shift = self._mapper.pose_shift(prev, f_clean) if prev is not None else None
        p_gate = self._gate(p_raw, shift)
        p_euro = self._euro(p_gate, t)

        # Projeção preditiva cinemática calculada sobre p_euro com amortecimento de frenagem (D4):
        v_lead = np.zeros(2, dtype=float)
        v_thresh = getattr(flt, "saccade_velocity_threshold", 80.0)
        inst_vel = np.zeros(2, dtype=float)
        speed_filt = 0.0
        if getattr(flt, "kinematic_prediction", False):
            if self._last_euro_p is not None and self._last_euro_t is not None:
                dt = t - self._last_euro_t
                if 1e-4 <= dt <= 0.5:
                    inst_vel = (p_euro - self._last_euro_p) / dt
                else:
                    inst_vel = np.zeros(2, dtype=float)
            else:
                inst_vel = np.zeros(2, dtype=float)

            speed_inst = float(np.linalg.norm(inst_vel))
            speed_filt = float(np.linalg.norm(self._vel_filtered))
            dot = float(np.dot(inst_vel, self._vel_filtered))

            # Amortecimento de frenagem (Braking Gate): zera overshoot e recuo elástico
            if speed_inst < speed_filt or dot <= 0.0:
                brake_factor = float(np.clip(speed_inst / max(speed_filt, 1.0), 0.0, 1.0))
                self._vel_filtered = inst_vel.copy()
                speed_filt = speed_inst
            else:
                self._vel_filtered = 0.6 * inst_vel + 0.4 * self._vel_filtered
                speed_filt = float(np.linalg.norm(self._vel_filtered))
                brake_factor = 1.0

            self._last_euro_p = p_euro.copy()
            self._last_euro_t = t

            if speed_filt > v_thresh:
                v_full = v_thresh * 3.0
                lead_s = getattr(flt, "lead_time_s", 0.020)
                w = float(np.clip((speed_filt - v_thresh) / max(v_full - v_thresh, 1.0), 0.0, 1.0)) * brake_factor
                v_lead = self._vel_filtered * (lead_s * w)
                self._saccade_dir = self._vel_filtered / max(speed_filt, 1e-6)
        else:
            self._last_euro_p = p_euro.copy()
            self._last_euro_t = t

        self._stabilizer.radius_px = self._fix_radius * (flt.edge_deadzone_scale if self._near_edge(p_euro) else 1.0)
        p = self._stabilizer(p_euro, t)
        p_dispatch = p + v_lead

        # Retenção monotônica sem recuo elástico ao desacelerar/pousar no alvo
        if self._saccade_dir is not None and self._last_dispatch is not None:
            is_reversal = (
                float(np.dot(inst_vel, self._saccade_dir)) < -5.0
                or float(np.dot(p_euro - self._last_dispatch, self._saccade_dir)) < -0.5 * self._fix_radius
            )
            if is_reversal:
                self._saccade_dir = None
            else:
                step_along_dir = float(np.dot(p_dispatch - self._last_dispatch, self._saccade_dir))
                if step_along_dir < 0.0:
                    p_dispatch = self._last_dispatch.copy()
                if speed_filt <= v_thresh:
                    if float(np.dot(p - self._last_dispatch, self._saccade_dir)) >= 0.0:
                        self._saccade_dir = None
                    else:
                        if float(np.linalg.norm(p - p_dispatch)) <= self._fix_radius:
                            self._stabilizer.center = p_dispatch.copy()
                        self._saccade_dir = None

        w_screen, h_screen = self._mapper.screen_size
        p_dispatch = np.array([
            np.clip(p_dispatch[0], 0.0, float(w_screen - 1)),
            np.clip(p_dispatch[1], 0.0, float(h_screen - 1)),
        ], dtype=float)

        self._last_dispatch = p_dispatch.copy()
        self._cursor.move_to(p_dispatch)
        self._history.push(t, p)

        # Atualiza a máquina de estados da Lupa 2.0
        mag_status = self._magnifier.update(self._stabilizer, p, t) if self._magnifier is not None else None

        # 1. Clique desacoplado por permanência dentro da Lupa Inteligente (Aim Dwell):
        if mag_status is not None and mag_status.fired_click and event is None:
            if mag_status.real_target is not None:
                self._cursor.move_to(np.array(mag_status.real_target, dtype=float))
            self._cursor.click("left")
            self._magnifier.dismiss()
            event = GestureEvent.DWELL_CLICK

        # 2. Clique por permanência na tela principal (suspenso quando a lupa estiver ativa):
        if mode in ("dwell", "both"):
            if self._magnifier is not None and self._magnifier.state is MagnifierState.ACTIVE:
                dwell_progress = mag_status.aim_progress if mag_status is not None else 0.0
            else:
                fired, dwell_progress = self._dwell.update(self._stabilizer, t)
                if fired and event is None:
                    self._cursor.click("left")
                    event = GestureEvent.DWELL_CLICK

        return SessionStatus(self._state, dwell_progress, event, hint, magnifier=mag_status)

    # ----------------------------------------------------------- helpers
    def _head_rate_deg_s(self, f: GazeFeatures) -> float:
        if self._prev is None:
            return 0.0
        dt = f.timestamp - self._prev.timestamp
        if dt <= 1e-3:
            return 0.0
        return math.degrees(math.hypot(f.yaw - self._prev.yaw, f.pitch - self._prev.pitch)) / dt

    def _update_noise(self, p_raw: np.ndarray) -> None:
        """One Euro adaptativo: filtra ruído fixacional (ignora sacadas) com suavização EMA."""
        flt = self._s.filters
        if self._last_raw is not None:
            delta = float(np.linalg.norm(p_raw - self._last_raw))
            # Ignora saltos de sacada (> 0.5 * jump_px)
            jump_lim = 0.5 * (flt.jump_deg * deg_to_px(1.0, DEFAULT_PITCH_MM, flt.viewing_distance_mm) if flt.jump_deg > 0 else flt.jump_px)
            if delta <= jump_lim:
                self._jitter.append(delta)
        self._last_raw = p_raw
        if flt.adaptive_noise and len(self._jitter) >= 8:
            noise = float(np.median(self._jitter))
            self._noise_ema = noise if self._noise_ema is None else (0.90 * self._noise_ema + 0.10 * noise)
            factor = float(np.clip(self._noise_ema / self._noise_ref_px, 1.0, 3.0))
            self._euro.set_min_cutoff(flt.min_cutoff / factor)

    def _near_edge(self, p: np.ndarray) -> bool:
        w, h = self._mapper.screen_size
        fr = self._s.filters.edge_frac
        return bool(p[0] < fr * w or p[0] > (1 - fr) * w or p[1] < fr * h or p[1] > (1 - fr) * h)

    def _light_hint(self, f: GazeFeatures, t: float) -> bool:
        if not self._cal_luma or f.face_luma <= 0:
            return False
        self._luma_ema = f.face_luma if self._luma_ema is None else self._luma_ema + 0.05 * (f.face_luma - self._luma_ema)
        tr = self._s.tracking
        if abs(self._luma_ema - self._cal_luma) / self._cal_luma > tr.luma_drift_frac:
            if self._drift_since is None:
                self._drift_since = t
            return t - self._drift_since >= tr.luma_drift_s
        self._drift_since = None
        return False

