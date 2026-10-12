"""Grava sessão de vídeo com timestamps para testes de regressão determinísticos."""
from __future__ import annotations

import json
import time
from pathlib import Path
import cv2

from lorena import paths
from lorena.config import Settings
from lorena.vision.camera import Camera


def record_session(output_dir: str = "sessions", duration_s: float = 10.0) -> None:
    settings = Settings.load(paths.CONFIG_FILE)
    camera = Camera(settings.camera)
    camera.open()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    session_id = time.strftime("%Y%m%d_%H%M%S")
    video_path = out_path / f"session_{session_id}.avi"
    meta_path = out_path / f"session_{session_id}.json"

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(video_path), fourcc, settings.camera.fps, (settings.camera.width, settings.camera.height))

    timestamps: list[float] = []
    print(f"Gravando sessão ({duration_s}s) em {video_path}...")
    start_t = time.monotonic()

    try:
        while time.monotonic() - start_t < duration_s:
            frame = camera.read()
            if frame is None:
                continue
            t = time.monotonic()
            timestamps.append(t)
            writer.write(frame)
    finally:
        camera.release()
        writer.release()

    meta = {
        "session_id": session_id,
        "width": settings.camera.width,
        "height": settings.camera.height,
        "fps": settings.camera.fps,
        "frames_recorded": len(timestamps),
        "timestamps": timestamps,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Sessão finalizada com {len(timestamps)} frames salvos.")


if __name__ == "__main__":
    record_session()
