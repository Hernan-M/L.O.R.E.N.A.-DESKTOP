"""Ferramenta de depuração visual do rastreio ocular do L.O.R.E.N.A.

Exibe em tempo real a detecção dos landmarks de íris, métricas de EAR,
razões de olhar (h, v), pose da cabeça e taxa de quadros (FPS).
Pressione ESC para encerrar.
"""
from __future__ import annotations

import time
import cv2
import numpy as np

from lorena import paths
from lorena.config import Settings
from lorena.vision.camera import Camera
from lorena.vision.face_detector import FaceLandmarkDetector
from lorena.vision.gaze_features import GazeFeatureExtractor, OpennessModel


def run_debug() -> None:
    settings = Settings.load(paths.CONFIG_FILE)
    camera = Camera(settings.camera)
    detector = FaceLandmarkDetector(paths.FACE_LANDMARKER_MODEL, settings.detector)
    openness = OpennessModel(settings.blink)
    extractor = GazeFeatureExtractor(openness)

    camera.open()
    cv2.namedWindow("L.O.R.E.N.A. - Visao Computacional (Debug)", cv2.WINDOW_NORMAL)

    prev_t = time.monotonic()
    fps = 0.0

    try:
        while True:
            frame = camera.read()
            if frame is None:
                continue

            now = time.monotonic()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_t, 1e-3))
            prev_t = now

            vis = cv2.flip(frame, 1)  # espelhado para pré-visualização amigável
            face = detector.detect(frame, now)  # detecção no frame original

            if face is not None:
                features = extractor.extract(face)
                fw = frame.shape[1]

                # Desenha os olhos no frame espelhado: x_vis = fw - 1 - x_orig
                for eye, label in ((face.left_eye, "L"), (face.right_eye, "R")):
                    # Íris
                    ix, iy = int(round(fw - 1 - eye.iris[0])), int(round(eye.iris[1]))
                    cv2.circle(vis, (ix, iy), 3, (0, 255, 255), -1)

                    # Cantos
                    sx, sy = int(round(fw - 1 - eye.start_corner[0])), int(round(eye.start_corner[1]))
                    ex, ey = int(round(fw - 1 - eye.end_corner[0])), int(round(eye.end_corner[1]))
                    cv2.circle(vis, (sx, sy), 2, (0, 255, 0), -1)
                    cv2.circle(vis, (ex, ey), 2, (0, 0, 255), -1)
                    cv2.line(vis, (sx, sy), (ex, ey), (255, 0, 0), 1)

                # Informações textuais na tela
                if features is not None:
                    txt1 = f"FPS: {fps:.1f}  |  Qualidade: {features.quality:.2f}"
                    txt2 = f"Olhar: h={features.h:.3f}, v={features.v:.3f}"
                    txt3 = f"Abertura: E={features.openness_left:.2f}, D={features.openness_right:.2f}"
                    txt4 = f"EAR: E={features.ear_left:.3f}, D={features.ear_right:.3f}"
                    txt5 = f"Pose: yaw={np.degrees(features.yaw):.1f}°, pitch={np.degrees(features.pitch):.1f}°"

                    y = 30
                    for line in (txt1, txt2, txt3, txt4, txt5):
                        cv2.putText(vis, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                        cv2.putText(vis, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 1, cv2.LINE_AA)
                        y += 26
            else:
                cv2.putText(vis, "Rosto nao detectado", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("L.O.R.E.N.A. - Visao Computacional (Debug)", vis)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_debug()
