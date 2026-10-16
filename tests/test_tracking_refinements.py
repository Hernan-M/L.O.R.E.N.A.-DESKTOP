import math
import numpy as np
import pytest

from lorena.config import BlinkSettings, Settings
from lorena.domain.models import CalibrationSample, EyeObservation, FaceObservation, GazeFeatures
from lorena.mapping.filters import FixationStabilizer, OutlierGate
from lorena.mapping.gaze_mapper import GazeMapper
from lorena.services.accuracy_evaluator import AccuracyEvaluator
from lorena.services.cursor import RecordingCursor
from lorena.services.cursor_session import CursorSession
from lorena.vision import landmarks as lm
from lorena.vision.face_detector import enhance_contrast_adaptive, face_from_landmarks, head_pose_from_matrix
from lorena.vision.gaze_features import GazeFeatureExtractor, OpennessModel


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
def test_binocular_vergence_invariance_and_wink_transition():
    settings = BlinkSettings(default_open_ear=0.30, closed_ratio=0.35)
    model = OpennessModel(settings, open_left=0.30, open_right=0.30)
    extractor = GazeFeatureExtractor(model)

    # Olhos com disparidade de vergência anatômica (Δh = 0.05)
    # Esquerdo em h=0.55, Direito em h=0.50 -> Ciclópico alvo = 0.525
    start_l = np.array([100.0, 100.0])
    end_l = np.array([140.0, 100.0])
    iris_l = np.array([122.0, 100.0])

    start_r = np.array([200.0, 100.0])
    end_r = np.array([240.0, 100.0])
    iris_r = np.array([220.0, 100.0])

    # Ambos os olhos bem abertos (EAR=0.30 -> openness=1.0 >= 0.75) para estabilizar EMA
    for _ in range(30):
        left_eye = EyeObservation(iris=iris_l, start_corner=start_l, end_corner=end_l, ear=0.30)
        right_eye = EyeObservation(iris=iris_r, start_corner=start_r, end_corner=end_r, ear=0.30)
        face = FaceObservation(timestamp=1.0, left_eye=left_eye, right_eye=right_eye)
        f = extractor.extract(face)

    assert f is not None
    assert pytest.approx(f.h, abs=1e-3) == 0.525

    # Simula transição progressiva de wink no olho esquerdo (openness cai de 1.0 a 0.0)
    h_positions = []
    ears = np.linspace(0.30, 0.10, 20)
    for ear_l in ears:
        left_eye = EyeObservation(iris=iris_l, start_corner=start_l, end_corner=end_l, ear=ear_l)
        right_eye = EyeObservation(iris=iris_r, start_corner=start_r, end_corner=end_r, ear=0.30)
        face = FaceObservation(timestamp=1.0, left_eye=left_eye, right_eye=right_eye)
        f_wink = extractor.extract(face)
        assert f_wink is not None
        h_positions.append(f_wink.h)

    # Sem descontinuidade ou salto brusco de vergência
    max_jump = max(h_positions) - min(h_positions)
    assert max_jump < 0.005


@pytest.mark.unit
def test_multi_landmark_corner_and_pose_scale_invariance():
    # 1. Cluster de 3 pontos para cantos (Mc = 3)
    np.random.seed(42)
    pts = np.zeros((478, 2))
    base_pos = np.array([80.0, 50.0])
    for idx in lm.RIGHT_EYE_START_CLUSTER:
        pts[idx] = base_pos + np.random.normal(0, 1.0, 2)
    for idx in lm.RIGHT_EYE_END_CLUSTER:
        pts[idx] = np.array([120.0, 50.0]) + np.random.normal(0, 1.0, 2)
    for idx in lm.LEFT_EYE_START_CLUSTER:
        pts[idx] = np.array([200.0, 50.0])
    for idx in lm.LEFT_EYE_END_CLUSTER:
        pts[idx] = np.array([240.0, 50.0])
    pts[lm.IRIS_CENTERS[0]] = [100.0, 50.0]
    for idx in lm.IRIS_RING_A:
        pts[idx] = [100.0, 50.0]
    pts[lm.IRIS_CENTERS[1]] = [220.0, 50.0]
    for idx in lm.IRIS_RING_B:
        pts[idx] = [220.0, 50.0]

    obs = face_from_landmarks(pts, 1.0)
    assert obs is not None
    assert np.allclose(obs.right_eye.start_corner, pts[list(lm.RIGHT_EYE_START_CLUSTER)].mean(axis=0))

    # 2. Invariância de escala de atan2 em head_pose_from_matrix
    cy, sy = math.cos(0.3), math.sin(0.3)
    r_mat = np.array([
        [cy, 0.0, sy],
        [0.0, 1.0, 0.0],
        [-sy, 0.0, cy]
    ])
    m1 = np.eye(4)
    m1[:3, :3] = r_mat
    m2 = np.eye(4)
    m2[:3, :3] = r_mat * 3.5  # escala homogênea arbitrária

    yaw1, pitch1, roll1 = head_pose_from_matrix(m1)
    yaw2, pitch2, roll2 = head_pose_from_matrix(m2)

    assert pytest.approx(yaw1, abs=1e-6) == 0.3
    assert pytest.approx(yaw2, abs=1e-6) == 0.3
    assert pytest.approx(pitch1, abs=1e-6) == 0.0
    assert pytest.approx(pitch2, abs=1e-6) == 0.0
    assert pytest.approx(roll1, abs=1e-6) == 0.0
    assert pytest.approx(roll2, abs=1e-6) == 0.0


@pytest.mark.unit
def test_mapper_centroid_alternating_refinement_and_conditioning():
    # 14 pontos (9 repouso + 5 rotação ativa)
    xs = [100.0, 960.0, 1820.0]
    ys = [100.0, 540.0, 980.0]
    targets = [(x, y) for y in ys for x in xs]
    targets += [(960.0, 540.0), (100.0, 100.0), (1820.0, 100.0), (100.0, 980.0), (1820.0, 980.0)]

    samples = []
    for idx, (tx, ty) in enumerate(targets):
        # Cantos do Passe 1 (idx in (0, 2, 6, 8)) com leve giro natural de cabeça (0.04 rad)
        if idx in (0, 2, 6, 8):
            yaw = 0.04
            pitch = 0.04
            hx = 0.01
        elif idx >= 9:
            yaw = 0.08
            pitch = 0.08
            hx = 0.04
        else:
            yaw = 0.008
            pitch = 0.008
            hx = 0.002
        h = 0.3 + 0.4 * (tx - 100.0) / 1720.0 - 0.2 * yaw
        v = -0.2 + 0.4 * (ty - 100.0) / 880.0 - 0.2 * pitch
        lid_val = 0.04 + 0.05 * v
        for _ in range(12):
            samples.append(
                CalibrationSample(
                    h=h + np.random.normal(0, 0.002),
                    v=v + np.random.normal(0, 0.002),
                    yaw=yaw + np.random.normal(0, 0.001),
                    pitch=pitch + np.random.normal(0, 0.001),
                    target_x=tx,
                    target_y=ty,
                    point_index=idx,
                    lid=lid_val + np.random.normal(0, 0.0005),
                    head_x=hx,
                    head_y=0.0,
                )
            )

    mapper = GazeMapper(degree=1, use_head_pose=True, use_lid=False, auto_select=False, screen_size=(1920, 1080))
    report = mapper.fit(samples)
    assert mapper.is_fitted
    assert report.loo_mean_error_px is not None
    assert report.loo_mean_error_px < 15.0

    beta = mapper._beta
    assert beta is not None
    assert beta[4, 0] == pytest.approx(0.0)
    assert beta[3, 1] == pytest.approx(0.0)

    # Teste de GazeMapper(degree=2, use_head_pose=True, use_lid=True)
    mapper2 = GazeMapper(degree=2, use_head_pose=True, use_lid=True, auto_select=False, screen_size=(1920, 1080))
    report2 = mapper2.fit(samples)
    assert mapper2.is_fitted
    assert mapper2.basis == "symmetric"
    assert mapper2.basis != "quadratic"
    assert report2.loo_mean_error_px is not None
    assert report2.loo_mean_error_px < 25.0

    mapper_auto = GazeMapper(degree=2, use_head_pose=True, use_lid=True, auto_select=True, screen_size=(1920, 1080))
    report_auto = mapper_auto.fit(samples)
    assert mapper_auto.is_fitted
    assert mapper_auto.basis != "quadratic"
    assert report_auto.loo_mean_error_px is not None
    assert report_auto.loo_mean_error_px < 25.0


@pytest.mark.integration
def test_no_elastic_whip_and_braking_damping():
    settings = Settings()
    settings.filters.kinematic_prediction = True
    settings.filters.saccade_velocity_threshold = 80.0
    settings.filters.lead_time_s = 0.020

    mapper = GazeMapper(degree=1, screen_size=(1920, 1080))
    mapper._mu = np.array([0.5, 0.0])
    mapper._sigma = np.array([0.1, 0.1])
    mapper._beta = np.array([[960.0, 540.0], [400.0, 0.0], [0.0, 400.0]])
    mapper._zmin = np.array([-2.0, -2.0])
    mapper._zmax = np.array([2.0, 2.0])

    cursor = RecordingCursor()
    session = CursorSession(mapper, cursor, settings)

    t = 0.0
    dt = 0.033

    # Repouso no centro (h = 0.50 -> 960 px)
    f_rest = GazeFeatures(t, 0.5, 0.0, 0.0, 0.0, 0.3, 0.3, 1.0, 1.0, 1.0)
    for _ in range(10):
        session.on_frame(f_rest, t)
        t += dt

    # Sacada rápida em direção a h = 0.60 (1360 px)
    dispatched = []
    for h in [0.53, 0.56, 0.59]:
        f_saccade = GazeFeatures(t, h, 0.0, 0.0, 0.0, 0.3, 0.3, 1.0, 1.0, 1.0)
        session.on_frame(f_saccade, t)
        dispatched.append(cursor.positions[-1][0])
        t += dt

    # Desaceleração súbita no pouso em h = 0.60
    f_land = GazeFeatures(t, 0.60, 0.0, 0.0, 0.0, 0.3, 0.3, 1.0, 1.0, 1.0)
    for _ in range(5):
        session.on_frame(f_land, t)
        dispatched.append(cursor.positions[-1][0])
        t += dt

    # Verifica que o cursor se move monotonamente sem recuo elástico
    diffs = np.diff(dispatched)
    assert np.all(diffs >= -1e-6)

    # Reversão suave do olhar na direção oposta (h = 0.58, 0.56, 0.54, 0.52, 0.50)
    # a velocidade moderada (< 80 px/s): 4 quadros (~132 ms) por passo dh = -0.02
    return_dispatched = []
    for h_rev in [0.58, 0.56, 0.54, 0.52, 0.50]:
        for _ in range(4):
            f_rev = GazeFeatures(t, h_rev, 0.0, 0.0, 0.0, 0.3, 0.3, 1.0, 1.0, 1.0)
            session.on_frame(f_rev, t)
            t += dt
        return_dispatched.append(cursor.positions[-1][0])

    # Comprova que o cursor acompanha o retorno para a esquerda sem travar em 1360 px
    assert return_dispatched[-1] < 1000.0
    assert return_dispatched[-1] < return_dispatched[0] < dispatched[-1]
    assert not any(pos >= dispatched[-1] for pos in return_dispatched)


@pytest.mark.unit
def test_c0_fixation_stabilizer_and_dwell_anchor():
    r_val = 40.0
    stabilizer = FixationStabilizer(radius_px=r_val, inner_ratio=0.5)

    c0 = np.array([500.0, 500.0])
    res0 = stabilizer(c0, 0.0)
    assert np.allclose(res0, c0)

    # 1. Dentro da zona morta interna (d = 10 px <= 20 px)
    p_inner = np.array([510.0, 500.0])
    res_inner = stabilizer(p_inner, 0.1)
    assert np.allclose(res_inner, c0)

    # 2. Exatamente na fronteira interna (d = 20 px)
    p_b_in = np.array([520.0, 500.0])
    res_b_in = stabilizer(p_b_in, 0.2)
    assert np.allclose(res_b_in, c0)

    # 3. Transição com ganho quadrático s^2 sobre o excesso (d = 30 px -> s = 0.5, delta = 0.25 * 10 = 2.5 px -> c_new = 502.5)
    stabilizer.reset()
    stabilizer(c0, 0.0)
    p_mid = np.array([530.0, 500.0])
    res_mid = stabilizer(p_mid, 0.3)
    assert pytest.approx(res_mid[0], abs=1e-4) == 502.5
    assert pytest.approx(res_mid[1], abs=1e-4) == 500.0

    # 4. Fronteira externa (d = 40 px a partir do centro 500 -> delta = 20 px -> c_new = 520.0)
    stabilizer.reset()
    stabilizer(c0, 0.0)
    p_b_out = np.array([540.0, 500.0])
    res_b_out = stabilizer(p_b_out, 0.4)
    assert pytest.approx(res_b_out[0], abs=1e-4) == 520.0

    # Continuidade C0 com d > R (d = 40.001 px a partir do centro 500 -> p - 20 = 520.001)
    stabilizer.reset()
    stabilizer(c0, 0.0)
    p_just_out = np.array([540.001, 500.0])
    res_just_out = stabilizer(p_just_out, 0.5)
    assert pytest.approx(res_just_out[0], abs=1e-3) == 520.001

    # 5. Dwell timer ancorado em fixation_start = 0.0
    assert stabilizer.fixation_start == 0.0
    assert pytest.approx(stabilizer.fixation_duration(0.5), abs=1e-4) == 0.5

    # Salto para longe (> R) cria nova âncora
    p_saccade = np.array([700.0, 500.0])
    stabilizer(p_saccade, 1.0)
    assert stabilizer.fixation_start == 1.0


@pytest.mark.integration
def test_pose_one_euro_and_outlier_gate_no_ratchet_drift():
    # 1. OutlierGate apply_shift sem reset destrutivo
    gate = OutlierGate(jump_px=100.0, confirm_frames=3)
    p0 = np.array([500.0, 500.0])
    gate(p0)

    gate.apply_shift(np.array([20.0, 0.0]))
    assert np.allclose(gate._ref, [520.0, 500.0])

    p_spike = np.array([820.0, 500.0])
    res = gate(p_spike)
    assert np.allclose(res, [520.0, 500.0])
    assert np.allclose(gate._ref, [520.0, 500.0])

    # 2. Pré-filtro OneEuro reduz ruído de pose em repouso
    settings = Settings()
    mapper = GazeMapper(degree=1, screen_size=(1920, 1080))
    mapper._mu = np.array([0.5, 0.0])
    mapper._sigma = np.array([0.1, 0.1])
    mapper._beta = np.array([[960.0, 540.0], [1000.0, 0.0], [0.0, 1000.0]])
    cursor = RecordingCursor()
    session = CursorSession(mapper, cursor, settings)

    t = 0.0
    dt = 0.033
    yaws_in = []
    for i in range(30):
        yaw_noise = 0.01 * (1.0 if i % 2 == 0 else -1.0)
        yaws_in.append(yaw_noise)
        f = GazeFeatures(t, 0.5, 0.0, yaw_noise, 0.0, 0.3, 0.3, 1.0, 1.0, 1.0)
        session.on_frame(f, t)
        t += dt

    yaw_filtered = session._euro_yaw._x
    assert abs(yaw_filtered) < np.std(yaws_in) * 0.35


@pytest.mark.integration
def test_rapid_micro_saccades_bidirectional_responsiveness():
    settings = Settings()
    settings.filters.kinematic_prediction = True
    settings.filters.saccade_velocity_threshold = 80.0
    settings.filters.lead_time_s = 0.020

    mapper = GazeMapper(degree=1, screen_size=(1920, 1080))
    mapper._mu = np.array([0.5, 0.0])
    mapper._sigma = np.array([0.1, 0.1])
    mapper._beta = np.array([[960.0, 540.0], [400.0, 0.0], [0.0, 400.0]])
    mapper._zmin = np.array([-2.0, -2.0])
    mapper._zmax = np.array([2.0, 2.0])

    cursor = RecordingCursor()
    session = CursorSession(mapper, cursor, settings)

    t = 0.0
    dt = 0.033

    # Estabilização inicial em h = 0.50 (960 px)
    f0 = GazeFeatures(t, 0.5, 0.0, 0.0, 0.0, 0.3, 0.3, 1.0, 1.0, 1.0)
    for _ in range(5):
        session.on_frame(f0, t)
        t += dt

    # Sacada em voo para a direita (h = 0.52 -> 0.54, dentro de jump_px=180 px)
    for h_step in [0.52, 0.54]:
        f_right = GazeFeatures(t, h_step, 0.0, 0.0, 0.0, 0.3, 0.3, 1.0, 1.0, 1.0)
        session.on_frame(f_right, t)
        t += dt
    pos_right = cursor.positions[-1][0]
    assert pos_right > 1000.0
    assert session._saccade_dir is not None and session._saccade_dir[0] > 0.0

    # Inversão imediata em pleno voo: sacada rápida para a esquerda (h = 0.46 -> 800 px)
    for _ in range(3):
        f_left = GazeFeatures(t, 0.46, 0.0, 0.0, 0.0, 0.3, 0.3, 1.0, 1.0, 1.0)
        session.on_frame(f_left, t)
        t += dt
    pos_left = cursor.positions[-1][0]
    # Comprova que a inversão imediata destravou o cursor e cruzou para a esquerda da origem (960 px)
    assert pos_left < 960.0
    assert pos_left < pos_right


