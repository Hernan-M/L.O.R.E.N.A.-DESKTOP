"""Configurações tipadas, carregadas de ``settings/config.json`` (opcional).

Chaves ausentes usam o padrão; chaves desconhecidas são ignoradas.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal

log = logging.getLogger(__name__)


@dataclass
class CameraSettings:
    index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30


@dataclass
class DetectorSettings:
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    use_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_lum_threshold: float = 110.0


@dataclass
class CalibrationSettings:
    margin_px: int = 80            # distância dos pontos até a borda da tela
    grid: int = 3                  # 3 => 9 pontos
    settle_s: float = 0.8          # tempo descartado para o olho chegar ao alvo
    collect_s: float = 1.2         # tempo de coleta válida por ponto
    min_samples: int = 12          # amostras mínimas após rejeição de outliers
    max_spread: float = 0.05       # dispersão robusta máxima (unidades de h/v)
    max_retries: int = 2
    min_quality: float = 0.5
    polynomial_degree: int = 2
    ridge_lambda: float = 1e-3
    use_head_pose: bool = True


@dataclass
class FilterSettings:
    min_cutoff: float = 0.8        # One Euro: menor = mais suave parado
    beta: float = 0.004            # One Euro: maior = menos atraso em sacadas
    d_cutoff: float = 1.0
    jump_px: float = 260.0         # salto considerado suspeito (outlier gate)
    confirm_frames: int = 3        # frames consistentes para aceitar um salto
    fixation_radius_px: float = 45.0


@dataclass
class BlinkSettings:
    click_eye: Literal["left", "right"] = "left"
    close_threshold: float = 0.30  # abertura normalizada abaixo disso = fechado
    reopen_threshold: float = 0.50 # histerese: acima disso = aberto de novo
    other_open_threshold: float = 0.45
    closed_ratio: float = 0.35     # EAR fechado ≈ closed_ratio * EAR aberto
    default_open_ear: float = 0.28
    min_wink_s: float = 0.20
    max_click_s: float = 1.00
    max_long_wink_s: float = 2.50
    pause_hold_s: float = 1.20
    refractory_s: float = 0.40
    freeze_rollback_s: float = 0.15


@dataclass
class TrackingSettings:
    click_mode: Literal["wink", "dwell", "both"] = "wink"
    dwell_s: float = 1.2
    lost_after_s: float = 0.5
    degraded_quality: float = 0.6


@dataclass
class Settings:
    camera: CameraSettings = field(default_factory=CameraSettings)
    detector: DetectorSettings = field(default_factory=DetectorSettings)
    calibration: CalibrationSettings = field(default_factory=CalibrationSettings)
    filters: FilterSettings = field(default_factory=FilterSettings)
    blink: BlinkSettings = field(default_factory=BlinkSettings)
    tracking: TrackingSettings = field(default_factory=TrackingSettings)
    profile: str = "default"

    @classmethod
    def load(cls, path: Path) -> "Settings":
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return _from_dict(cls, data)
        except (OSError, ValueError, TypeError) as exc:
            log.warning("Configuração inválida em %s (%s); usando padrões.", path, exc)
            return cls()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")


def _from_dict(cls: type, data: dict[str, Any]):
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        default = f.default_factory() if callable(f.default_factory) else f.default  # type: ignore[misc]
        kwargs[f.name] = _from_dict(type(default), value) if is_dataclass(default) else value
    return cls(**kwargs)
