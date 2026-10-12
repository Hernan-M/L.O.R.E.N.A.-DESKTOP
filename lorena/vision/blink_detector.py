"""Detector de gestos de piscada (wink) com histerese e janelas de tempo.

Corrige os problemas do ``is_left_blinking`` antigo:
* não depende de achar a pupila (usa EAR normalizado);
* piscadas naturais (os dois olhos) são ignoradas;
* exige duração mínima (ruído de 1 frame não clica);
* período refratário (um gesto = um clique);
* expõe ``freeze`` para congelar o cursor enquanto a pálpebra fecha.

Estados::

    OPEN → WINK (olho de clique fecha, outro aberto) → CLICK/RIGHT_CLICK → REFRACTORY
    OPEN → BOTH (ambos fecham) → OPEN (piscada natural) | TOGGLE_PAUSE (longa)
"""
from __future__ import annotations

from enum import Enum

from lorena.config import BlinkSettings
from lorena.domain.models import GestureEvent


class _State(Enum):
    OPEN = "open"
    WINK = "wink"
    BOTH = "both"
    REFRACTORY = "refractory"


class BlinkDetector:
    def __init__(self, settings: BlinkSettings) -> None:
        self._s = settings
        self._state = _State.OPEN
        self._t0 = 0.0

    @property
    def state(self) -> str:
        return self._state.value

    @property
    def freeze(self) -> bool:
        """Verdadeiro enquanto um gesto está em andamento (cursor deve ficar parado)."""
        return self._state in (_State.WINK, _State.BOTH)

    def reset(self) -> None:
        self._state = _State.OPEN

    def update(self, openness_left: float | None, openness_right: float | None, t: float) -> GestureEvent | None:
        if openness_left is None or openness_right is None:
            self.reset()
            return None
        s = self._s
        click, other = (openness_left, openness_right) if s.click_eye == "left" else (openness_right, openness_left)
        click_closed, click_open = click < s.close_threshold, click > s.reopen_threshold
        other_closed, other_open = other < s.close_threshold, other > s.reopen_threshold

        if self._state is _State.OPEN:
            if click_closed and other_closed:
                self._enter(_State.BOTH, t)
            elif click_closed and other > s.other_open_threshold:
                self._enter(_State.WINK, t)
            return None

        if self._state is _State.WINK:
            if other_closed:
                self._state = _State.BOTH  # era uma piscada natural; mantém t0
                return None
            if click_open:
                duration = t - self._t0
                if duration < s.min_wink_s:
                    self._state = _State.OPEN
                    return None
                if duration <= s.max_click_s:
                    return self._fire(GestureEvent.CLICK, t)
                if duration <= s.max_long_wink_s:
                    return self._fire(GestureEvent.RIGHT_CLICK, t)
                self._state = _State.OPEN
            return None

        if self._state is _State.BOTH:
            if click_open and other_open:
                if t - self._t0 >= s.pause_hold_s:
                    return self._fire(GestureEvent.TOGGLE_PAUSE, t)
                self._state = _State.OPEN
            return None

        # REFRACTORY
        if t - self._t0 >= s.refractory_s and click_open and other_open:
            self._state = _State.OPEN
        return None

    def _enter(self, state: _State, t: float) -> None:
        self._state, self._t0 = state, t

    def _fire(self, event: GestureEvent, t: float) -> GestureEvent:
        self._enter(_State.REFRACTORY, t)
        return event
