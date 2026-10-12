"""Reproduz uma sessão gravada e avalia métricas do pipeline de visão."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import cv2
import numpy as np

from lorena import paths
from lorena.config import Settings
from lorena.domain.models import GazeFeatures
from lorena.vision.face_detector import FaceLandmarkDetector
from lorena.vision.gaze_features import GazeFeatureExtractor, OpennessModel


class VideoFileCamera:
    """Simula a interface Camera lendo frames de um arquivo de vídeo."""

    def __init__(self, video_path: str) -> None:
        self.video_path = video_path
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir o vídeo: {self.video_path}")

    def read(self) -> np.ndarray | None:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        return frame if ok and frame is not None else None

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def replay_session(video_path: str) -> dict:
    settings = Settings.load(paths.CONFIG_FILE)
    detector = FaceLandmarkDetector(paths.FACE_LANDMARKER_MODEL, settings.detector)
    openness = OpennessModel(settings.blink)
    extractor = GazeFeatureExtractor(openness)
    camera = VideoFileCamera(video_path)

    camera.open()
    total_frames = 0
    faces_detected = 0
    h_values: list[float] = []
    v_values: list[float] = []

    print(f"Processando replay: {video_path}...")
    try:
        t = 0.0
        while True:
            frame = camera.read()
            if frame is None:
                break
            total_frames += 1
            face = detector.detect(frame, t)
            if face is not None:
                faces_detected += 1
                feat = extractor.extract(face)
                if feat is not None and not np.isnan(feat.h):
                    h_values.append(feat.h)
                    v_values.append(feat.v)
            t += 1.0 / settings.camera.fps
    finally:
        camera.release()
        detector.close()

    tracking_ratio = (faces_detected / total_frames) if total_frames > 0 else 0.0
    jitter_h = float(np.std(np.diff(h_values))) if len(h_values) > 1 else 0.0
    jitter_v = float(np.std(np.diff(v_values))) if len(v_values) > 1 else 0.0

    results = {
        "total_frames": total_frames,
        "faces_detected": faces_detected,
        "tracking_ratio_pct": round(tracking_ratio * 100, 2),
        "jitter_rms_h": round(jitter_h, 5),
        "jitter_rms_v": round(jitter_v, 5),
    }

    print("\n--- Relatório de Replay ---")
    print(f"Total de quadros: {total_frames}")
    print(f"Quadros com rosto: {faces_detected} ({results['tracking_ratio_pct']}%)")
    print(f"Jitter delta H (RMS): {results['jitter_rms_h']}")
    print(f"Jitter delta V (RMS): {results['jitter_rms_v']}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay e benchmark de sessão gravada.")
    parser.add_argument("video", help="Caminho do arquivo de vídeo da sessão")
    args = parser.parse_args()
    replay_session(args.video)
