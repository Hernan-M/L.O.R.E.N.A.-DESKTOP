"""Sessão de controle do cursor: features → pixel filtrado → cursor + gestos.

Roda na thread de visão (baixa latência). Estados de rastreio:
TRACKING → DEGRADED (baixa confiança) → LOST (sem rosto > lost_after_s).
Em LOST o cursor fica parado (não "voa") e o filtro é reiniciado na volta.
"""
from __future__ import annotations

import logging
import math

import numpy as np

from lorena.config import Settings
from lorena.domain.models import GazeFeatures, GestureEvent, SessionStatus, TrackingState
from lorena.mapping.filters import DwellClicker, FixationStabilizer, OneEuroFilter2D, OutlierGate, PositionHistory
from lorena.mapping.gaze_mapper import GazeMapper
from lorena.services.cursor import Cursor
from lorena.vision.blink_detector import BlinkDetector

log = logging.getLogger(__name__)


class CursorSession:
    def __init__(self, mapper: GazeMapper, cursor: Cursor, settings: Settings) -> None:
        self._mapper = mapper
        self._cursor = cursor
        self._s = settings
        f = settings.filters
        self._gate = OutlierGate(f.jump_px, f.confirm_frames)
        self._euro = OneEuroFilter2D(f.min_cutoff, f.beta, f.d_cutoff)
        self._stabilizer = FixationStabilizer(f.fixation_radius_px)
        self._dwell = DwellClicker(settings.tracking.dwell_s)
        self._blink = BlinkDetector(settings.blink)
        self._history = PositionHistory()
        self._frozen_at: np.ndarray | None = None
        self._last_seen: float | None = None
        self._state = TrackingState.TRACKING
        self._paused = False

    @property
    def paused(self) -> bool:
        return self._paused

    def _reset_filters(self) -> None:
        self._gate.reset()
        self._euro.reset()
        self._stabilizer.reset()
        self._history.clear()

    def on_frame(self, f: GazeFeatures | None, t: float) -> SessionStatus:
        mode = self._s.tracking.click_mode
        event: GestureEvent | None = None
        dwell_progress = 0.0

        # --- presença do rosto ---------------------------------------------
        if f is None:
            self._blink.reset()
            self._frozen_at = None
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
            if self._frozen_at is not None:
                self._cursor.move_to(self._frozen_at)
            self._cursor.click("right" if event is GestureEvent.RIGHT_CLICK else "left")
        self._frozen_at = None

        # --- movimento -------------------------------------------------------
        if not (math.isfinite(f.h) and math.isfinite(f.v)):
            return SessionStatus(self._state, 0.0, event)
        was_lost = self._state is TrackingState.LOST
        self._state = TrackingState.DEGRADED if f.quality < self._s.tracking.degraded_quality else TrackingState.TRACKING
        if was_lost:
            self._reset_filters()

        p = self._mapper.predict(f)
        p = self._gate(p)
        p = self._euro(p, t)
        p = self._stabilizer(p, t)
        self._cursor.move_to(p)
        self._history.push(t, p)

        if mode in ("dwell", "both"):
            fired, dwell_progress = self._dwell.update(self._stabilizer, t)
            if fired and event is None:
                self._cursor.click("left")
                event = GestureEvent.DWELL_CLICK

        return SessionStatus(self._state, dwell_progress, event)
