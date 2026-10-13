import numpy as np
import pytest

from lorena.domain.models import CalibrationSample, GazeFeatures
from lorena.mapping.gaze_mapper import GazeMapper
from lorena.services.accuracy_evaluator import AccuracyEvaluator
from lorena.vision.face_detector import enhance_contrast_adaptive


def test_mapper_head_pose_fit_and_predict():
    mapper = GazeMapper(degree=2, ridge_lambda=1e-5, use_head_pose=True, screen_size=(1920, 1080))
    assert mapper.num_terms == 10  # 6 termos de grau 2 + 4 termos de pose (yaw, pitch, hx, hy)

    # 9 alvos na tela
    xs = [100.0, 960.0, 1820.0]
    ys = [100.0, 540.0, 980.0]
    targets = [(x, y) for y in ys for x in xs]

    samples = []
    for idx, (tx, ty) in enumerate(targets):
        h = 0.3 + 0.4 * (tx - 100.0) / 1720.0
        v = -0.2 + 0.4 * (ty - 100.0) / 880.0
        # Simula variação de pose da cabeça entre os pontos
        yaw = float(np.sin(idx)) * 0.1
        pitch = float(np.cos(idx)) * 0.1
        for _ in range(8):
            samples.append(
                CalibrationSample(
                    h=h + np.random.normal(0, 0.001),
                    v=v + np.random.normal(0, 0.001),
                    yaw=yaw,
                    pitch=pitch,
                    target_x=tx,
                    target_y=ty,
                    point_index=idx,
                )
            )

    report = mapper.fit(samples)
    assert mapper.is_fitted
    assert report.num_points == 9
    assert report.mean_error_px < 25.0

    # Testa predição com pose
    pred = mapper.predict_raw(0.5, 0.0, yaw=0.0, pitch=0.0)
    assert 0 <= pred[0] <= 1920
    assert 0 <= pred[1] <= 1080


def test_mapper_recalibrate_drift():
    mapper = GazeMapper(degree=1, use_head_pose=False, screen_size=(1920, 1080))
    mapper._mu = np.array([0.5, 0.0])
    mapper._sigma = np.array([0.1, 0.1])
    mapper._beta = np.array([[960.0, 540.0], [500.0, 0.0], [0.0, 300.0]])

    feat = GazeFeatures(
        timestamp=1.0,
        h=0.5,
        v=0.0,
        yaw=0.0,
        pitch=0.0,
        ear_left=0.3,
        ear_right=0.3,
        openness_left=0.9,
        openness_right=0.9,
        quality=1.0,
    )

    pred_before = mapper.predict(feat)
    np.testing.assert_allclose(pred_before, [960.0, 540.0], atol=1e-3)

    # Aplica recalibração de 1 ponto contra alvo (1000, 500)
    dx, dy = mapper.recalibrate_drift(feat, (1000.0, 500.0))
    assert pytest.approx(dx, abs=1e-2) == 40.0
    assert pytest.approx(dy, abs=1e-2) == -40.0

    pred_after = mapper.predict(feat)
    np.testing.assert_allclose(pred_after, [1000.0, 500.0], atol=1e-3)

    # Teste reset
    mapper.reset_drift_offset()
    np.testing.assert_allclose(mapper.predict(feat), [960.0, 540.0], atol=1e-3)


def test_clahe_adaptive_enhancement():
    # Imagem escura (luminância baixa) -> CLAHE deve ser aplicado e aumentar contraste
    dark_img = np.full((100, 100, 3), 40, dtype=np.uint8)
    enhanced = enhance_contrast_adaptive(dark_img, clip_limit=3.0, lum_threshold=110.0)
    assert enhanced.shape == dark_img.shape
    assert enhanced.dtype == np.uint8

    # Imagem clara (luminância acima de 110) -> deve ser retornada intacta
    bright_img = np.full((100, 100, 3), 180, dtype=np.uint8)
    not_modified = enhance_contrast_adaptive(bright_img, clip_limit=3.0, lum_threshold=110.0)
    assert np.array_equal(bright_img, not_modified)


def test_accuracy_evaluator():
    mapper = GazeMapper(degree=1, use_head_pose=False, screen_size=(1920, 1080))
    mapper._mu = np.array([0.5, 0.0])
    mapper._sigma = np.array([0.1, 0.1])
    mapper._beta = np.array([[960.0, 540.0], [500.0, 0.0], [0.0, 300.0]])

    evaluator = AccuracyEvaluator(mapper, screen_size=(1920, 1080), viewing_distance_mm=600.0, pixel_pitch_mm=0.27)

    # 100 pixels de erro equivalem a quantos graus?
    deg = evaluator.pixels_to_degrees(100.0)
    # arctan(27 / 600) * 180 / pi ≈ 2.576 graus
    assert 2.0 < deg < 3.0

    # Gera alvos aleatórios
    targets = evaluator.generate_random_targets(count=5, margin_px=80)
    assert len(targets) == 5
    for x, y in targets:
        assert 80 <= x <= 1920 - 80
        assert 80 <= y <= 1080 - 80

    # Avaliação com amostra sintética
    sample = GazeFeatures(
        timestamp=1.0,
        h=0.5,
        v=0.0,
        yaw=0.0,
        pitch=0.0,
        ear_left=0.3,
        ear_right=0.3,
        openness_left=0.9,
        openness_right=0.9,
        quality=1.0,
    )
    # Alvo no ponto previsto exato (960, 540)
    report = evaluator.evaluate_points([((960.0, 540.0), [sample, sample])])
    assert report.grade == "Excelente"
    assert pytest.approx(report.mean_error_px, abs=1e-3) == 0.0
    assert pytest.approx(report.mean_error_deg, abs=1e-3) == 0.0
