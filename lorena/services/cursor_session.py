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

import numpy as np

from lorena.config import Settings
from lorena.domain.models import GazeFeatures, GestureEvent, SessionStatus, TrackingState
from lorena.mapping.filters import DwellClicker, FixationStabilizer, OneEuroFilter2D, OutlierGate, PositionHistory
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
        # --- cabeça / ruído / luz ---
        self._prev: GazeFeatures | None = None
        self._hold_until = 0.0
        self._jitter: deque[float] = deque(maxlen=15)
        self._last_raw: np.ndarray | None = None
        self._cal_luma = calibration_luma
        self._luma_ema: float | None = None
        self._drift_since: float | None = None
        # --- projeção cinemática (Opção A) ---
        self._vel_filtered: np.ndarray = np.zeros(2, dtype=float)
        self._last_filt_p: np.ndarray | None = None
        self._last_filt_t: float | None = None

    @property
    def paused(self) -> bool:
        return self._paused

    def _reset_filters(self) -> None:
        self._gate.reset()
        self._euro.reset()
        self._stabilizer.reset()
        if self._magnifier is not None:
            self._magnifier.reset()
        self._history.clear()
        self._jitter.clear()
        self._last_raw = None
        self._prev = None
        self._hold_until = 0.0
        self._vel_filtered = np.zeros(2, dtype=float)
        self._last_filt_p = None
        self._last_filt_t = None

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

        # --- movimento -------------------------------------------------------
        if not (math.isfinite(f.h) and math.isfinite(f.v)):
            return SessionStatus(self._state, 0.0, event)
        was_lost = self._state is TrackingState.LOST
        out_of_range = self._mapper.pose_out_of_range(f)
        degraded = f.quality < self._s.tracking.degraded_quality or out_of_range
        self._state = TrackingState.DEGRADED if degraded else TrackingState.TRACKING
        if was_lost:
            self._reset_filters()
        hint = "head" if out_of_range else ("light" if self._light_hint(f, t) else None)

        # Giro rápido da cabeça: os landmarks ficam instáveis → congela por um instante.
        flt = self._s.filters
        if self._head_rate_deg_s(f) > flt.head_rate_deg_s:
            self._hold_until = t + flt.head_hold_s
        prev, self._prev = self._prev, f
        if t < self._hold_until:
            self._gate.reset()  # após o giro, o novo ponto é a nova referência
            self._vel_filtered = np.zeros(2, dtype=float)
            self._last_filt_p = None
            self._last_filt_t = None
            return SessionStatus(self._state, 0.0, event, hint)

        p_raw = self._mapper.predict(f)
        self._update_noise(p_raw)
        shift = self._mapper.pose_shift(prev, f) if prev is not None else None
        p = self._gate(p_raw, shift)
        p = self._euro(p, t)
        self._stabilizer.radius_px = self._fix_radius * (flt.edge_deadzone_scale if self._near_edge(p) else 1.0)
        p = self._stabilizer(p, t)

        # Projeção preditiva cinemática (Opção A):
        # Durante sacadas, projeta a posição do cursor ligeiramente à frente ao longo do vetor
        # de velocidade filtrado, compensando o atraso de ~20ms da exposição e captura USB.
        p_dispatch = p
        if getattr(flt, "kinematic_prediction", False):
            if self._last_filt_p is not None and self._last_filt_t is not None:
                dt = t - self._last_filt_t
                if 1e-4 <= dt <= 0.5:
                    inst_vel = (p - self._last_filt_p) / dt
                    self._vel_filtered = 0.6 * inst_vel + 0.4 * self._vel_filtered
                else:
                    self._vel_filtered = np.zeros(2, dtype=float)
            else:
                self._vel_filtered = np.zeros(2, dtype=float)
            self._last_filt_p = p.copy()
            self._last_filt_t = t

            speed = float(np.linalg.norm(self._vel_filtered))
            v_thresh = getattr(flt, "saccade_velocity_threshold", 80.0)
            if speed > v_thresh:
                v_full = v_thresh * 3.0
                lead_s = getattr(flt, "lead_time_s", 0.020)
                w = float(np.clip((speed - v_thresh) / max(v_full - v_thresh, 1.0), 0.0, 1.0))
                p_proj = p + self._vel_filtered * (lead_s * w)
                w_screen, h_screen = self._mapper.screen_size
                p_dispatch = np.array([
                    np.clip(p_proj[0], 0.0, float(w_screen - 1)),
                    np.clip(p_proj[1], 0.0, float(h_screen - 1)),
                ], dtype=float)
        else:
            self._last_filt_p = p.copy()
            self._last_filt_t = t

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
        """One Euro adaptativo: mais ruído (pouca luz) → corte menor → mais suavização."""
        flt = self._s.filters
        if self._last_raw is not None:
            self._jitter.append(float(np.linalg.norm(p_raw - self._last_raw)))
        self._last_raw = p_raw
        if flt.adaptive_noise and len(self._jitter) >= 8:
            noise = float(np.median(self._jitter))  # mediana ignora as sacadas reais
            factor = float(np.clip(noise / self._noise_ref_px, 1.0, 3.0))
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

