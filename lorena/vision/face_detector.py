"""Detecção de landmarks faciais via MediaPipe Tasks ``FaceLandmarker``."""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from lorena.config import DetectorSettings
from lorena.domain.models import EyeObservation, FaceObservation
from lorena.vision import landmarks as lm


def eye_aspect_ratio(p: np.ndarray) -> float:
    """EAR (Soukupová & Čech, 2016) a partir de 6 pontos (6, 2).

    EAR = (|p2-p6| + |p3-p5|) / (2 |p1-p4|). ~0.25–0.35 aberto, → 0 fechado.
    """
    width = np.linalg.norm(p[0] - p[3])
    if width < 1e-6:
        return 0.0
    height = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    return float(height / (2.0 * width))


def head_pose_from_matrix(matrix: np.ndarray | None) -> tuple[float, float, float]:
    """(yaw, pitch, roll) em radianos a partir da matriz 4x4 de transformação facial."""
    if matrix is None:
        return 0.0, 0.0, 0.0
    r = np.asarray(matrix, dtype=float)[:3, :3]
    pitch = math.atan2(r[2, 1], r[2, 2])
    yaw = math.asin(float(np.clip(-r[2, 0], -1.0, 1.0)))
    roll = math.atan2(r[1, 0], r[0, 0])
    return yaw, pitch, roll


def face_from_landmarks(points: np.ndarray, timestamp: float, matrix: np.ndarray | None = None) -> FaceObservation | None:
    """Monta um ``FaceObservation`` a partir de landmarks em pixels (N, 2).

    Função pura (testável sem câmera nem MediaPipe).
    """
    if points.shape[0] < lm.NUM_LANDMARKS_WITH_IRIS:
        return None

    def build(corners: tuple[int, int], ear_idx: tuple[int, ...], iris: np.ndarray) -> EyeObservation:
        return EyeObservation(
            iris=iris,
            start_corner=points[corners[0]].copy(),
            end_corner=points[corners[1]].copy(),
            ear=eye_aspect_ratio(points[list(ear_idx)]),
        )

    right_mid = (points[lm.RIGHT_EYE_CORNERS[0]] + points[lm.RIGHT_EYE_CORNERS[1]]) / 2
    left_mid = (points[lm.LEFT_EYE_CORNERS[0]] + points[lm.LEFT_EYE_CORNERS[1]]) / 2
    iris_a, iris_b = points[lm.IRIS_CENTERS[0]].copy(), points[lm.IRIS_CENTERS[1]].copy()
    # Atribuição defensiva por proximidade: independe da convenção de índices.
    straight = np.linalg.norm(iris_a - right_mid) + np.linalg.norm(iris_b - left_mid)
    swapped = np.linalg.norm(iris_a - left_mid) + np.linalg.norm(iris_b - right_mid)
    right_iris, left_iris = (iris_a, iris_b) if straight <= swapped else (iris_b, iris_a)

    yaw, pitch, roll = head_pose_from_matrix(matrix)
    return FaceObservation(
        timestamp=timestamp,
        left_eye=build(lm.LEFT_EYE_CORNERS, lm.LEFT_EYE_EAR, left_iris),
        right_eye=build(lm.RIGHT_EYE_CORNERS, lm.RIGHT_EYE_EAR, right_iris),
        yaw=yaw,
        pitch=pitch,
        roll=roll,
    )


def enhance_contrast_adaptive(frame_bgr: np.ndarray, clip_limit: float = 2.0, lum_threshold: float = 110.0) -> np.ndarray:
    """Aplica equalização CLAHE no canal L (LAB) se a iluminação estiver baixa."""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    if float(np.mean(l_channel)) < lum_threshold:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)
        lab = cv2.merge([l_enhanced, a_channel, b_channel])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return frame_bgr


class FaceLandmarkDetector:
    """Wrapper do ``FaceLandmarker`` em modo VIDEO (usa rastreio entre frames)."""

    def __init__(self, model_path: Path, settings: DetectorSettings) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        self._settings = settings
        # model_asset_buffer evita problemas do MediaPipe com caminhos não-ASCII/OneDrive.
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=Path(model_path).read_bytes()),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=settings.min_detection_confidence,
            min_face_presence_confidence=settings.min_presence_confidence,
            min_tracking_confidence=settings.min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def detect(self, frame_bgr: np.ndarray, timestamp: float) -> FaceObservation | None:
        ts_ms = max(int(timestamp * 1000), self._last_ts_ms + 1)  # precisa ser estritamente crescente
        self._last_ts_ms = ts_ms
        processed_bgr = (
            enhance_contrast_adaptive(
                frame_bgr,
                self._settings.clahe_clip_limit,
                self._settings.clahe_lum_threshold,
            )
            if self._settings.use_clahe
            else frame_bgr
        )
        rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, ts_ms)
        if not result.face_landmarks:
            return None
        h, w = frame_bgr.shape[:2]
        points = np.array([(p.x * w, p.y * h) for p in result.face_landmarks[0]], dtype=np.float64)
        matrix = result.facial_transformation_matrixes[0] if result.facial_transformation_matrixes else None
        return face_from_landmarks(points, timestamp, matrix)

    def close(self) -> None:
        self._landmarker.close()
