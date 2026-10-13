"""Conversões entre ângulo visual e pixels, usando o tamanho físico da tela."""
from __future__ import annotations

import math

DEFAULT_PITCH_MM = 25.4 / 96.0  # 96 DPI ≈ 0,2646 mm/px (fallback quando o SO não informa)


def deg_to_px(deg: float, pitch_mm: float, distance_mm: float = 600.0) -> float:
    """Pixels equivalentes a ``deg`` graus de ângulo visual (no centro, aproximação tan)."""
    return distance_mm * math.tan(math.radians(deg)) / max(pitch_mm, 1e-6)


def sane_pitch(pitch_mm: float | None) -> float:
    """Descarta valores absurdos reportados pelo driver (faixa plausível: 0,10–0,50 mm/px)."""
    if pitch_mm is None or not (0.10 <= pitch_mm <= 0.50):
        return DEFAULT_PITCH_MM
    return float(pitch_mm)
