"""Persistência do perfil de calibração (JSON versionado)."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from lorena.mapping.gaze_mapper import CalibrationReport, GazeMapper

log = logging.getLogger(__name__)


@dataclass
class CalibrationProfile:
    mapper: GazeMapper
    open_ear_left: float
    open_ear_right: float
    screen_size: tuple[int, int]
    report: CalibrationReport
    created_at: float = field(default_factory=time.time)


class CalibrationRepository:
    VERSION = 1

    def __init__(self, directory: Path) -> None:
        self._dir = directory

    def _path(self, profile: str) -> Path:
        return self._dir / f"profile_{profile}.json"

    def save(self, name: str, profile: CalibrationProfile) -> Path:
        self._dir.mkdir(parents=True, exist_ok=True)
        data = {
            "version": self.VERSION,
            "created_at": profile.created_at,
            "screen_size": list(profile.screen_size),
            "open_ear": {"left": profile.open_ear_left, "right": profile.open_ear_right},
            "report": asdict(profile.report),
            "mapper": profile.mapper.to_dict(),
        }
        path = self._path(name)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(path)  # escrita atômica
        return path

    def load(self, name: str) -> CalibrationProfile | None:
        path = self._path(name)
        if not path.exists():
            return None
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            if d.get("version") != self.VERSION:
                raise ValueError("versão incompatível")
            report = d["report"]
            report["per_point_error_px"] = {int(k): v for k, v in report.get("per_point_error_px", {}).items()}
            return CalibrationProfile(
                mapper=GazeMapper.from_dict(d["mapper"]),
                open_ear_left=float(d["open_ear"]["left"]),
                open_ear_right=float(d["open_ear"]["right"]),
                screen_size=(int(d["screen_size"][0]), int(d["screen_size"][1])),
                report=CalibrationReport(**report),
                created_at=float(d.get("created_at", 0.0)),
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            log.warning("Perfil de calibração inválido em %s: %s", path, exc)
            return None
