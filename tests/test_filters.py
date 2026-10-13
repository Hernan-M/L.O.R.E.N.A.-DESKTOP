import numpy as np
import pytest

from lorena.mapping.filters import (
    DwellClicker,
    FixationStabilizer,
    OneEuroFilter,
    OneEuroFilter2D,
    OutlierGate,
    PositionHistory,
)


def test_one_euro_filter_reduces_jitter():
    # Low cutoff, small beta -> strong smoothing on stationary signal
    f = OneEuroFilter(min_cutoff=0.5, beta=0.001)

    t = 0.0
    dt = 0.033  # ~30 FPS
    values = [500.0 + (5.0 if i % 2 == 0 else -5.0) for i in range(30)]

    filtered = []
    for v in values:
        filtered.append(f(v, t))
        t += dt

    # Filtered variance should be much smaller than raw variance
    assert np.std(filtered[10:]) < np.std(values[10:]) * 0.4


def test_outlier_gate_rejects_single_spike():
    gate = OutlierGate(jump_px=100.0, confirm_frames=3)

    p0 = np.array([500.0, 500.0])
    assert np.allclose(gate(p0), p0)

    # Spike of 300px
    p_spike = np.array([800.0, 500.0])
    res_spike = gate(p_spike)
    # Should maintain previous stable reference
    assert np.allclose(res_spike, p0)

    # Returns to normal
    assert np.allclose(gate(p0), p0)


def test_outlier_gate_accepts_sustained_movement():
    gate = OutlierGate(jump_px=100.0, confirm_frames=3)

    p0 = np.array([500.0, 500.0])
    gate(p0)

    p_new = np.array([800.0, 500.0])
    # 1st frame of jump -> holds p0
    assert np.allclose(gate(p_new), p0)
    # 2nd frame of jump -> holds p0
    assert np.allclose(gate(p_new), p0)
    # 3rd frame of jump -> confirms jump to p_new
    res = gate(p_new)
    assert np.allclose(res, p_new)


def test_fixation_stabilizer_and_dwell():
    stabilizer = FixationStabilizer(radius_px=30.0)
    dwell = DwellClicker(dwell_s=0.5)

    p_center = np.array([500.0, 500.0])

    t = 0.0
    # First frame initializes
    res = stabilizer(p_center, t)
    assert np.allclose(res, p_center)

    # Micro-movements within 30px deadzone
    t = 0.2
    p_jitter = np.array([515.0, 505.0])
    res = stabilizer(p_jitter, t)
    assert np.allclose(res, p_center)  # Cursor kept anchored

    fired, progress = dwell.update(stabilizer, t)
    assert not fired
    assert 0.35 <= progress <= 0.45

    # Reach 0.6s of fixation -> dwell should trigger
    t = 0.6
    fired, progress = dwell.update(stabilizer, t)
    assert fired
    assert progress == 1.0


def test_position_history():
    hist = PositionHistory(maxlen=10)

    hist.push(1.0, np.array([100.0, 100.0]))
    hist.push(1.1, np.array([110.0, 110.0]))
    hist.push(1.2, np.array([120.0, 120.0]))

    # Retrieve position at t=1.05 -> should find point at t=1.0
    p = hist.at(1.05)
    assert p is not None
    assert np.allclose(p, np.array([100.0, 100.0]))


def test_one_euro_filter_2d_isotropic_rotational_invariance():
    # Modo isotrópico deve garantir simetria exata ao longo de diagonais
    f_iso = OneEuroFilter2D(min_cutoff=0.8, beta=0.005, d_cutoff=1.0, isotropic=True)

    t = 0.0
    dt = 0.0333  # ~30 FPS
    for i in range(20):
        # Movimento em 45 graus: x e y crescem juntos
        p_raw = np.array([100.0 + i * 15.0, 100.0 + i * 15.0])
        p_filt = f_iso(p_raw, t)
        assert pytest.approx(p_filt[0], abs=1e-5) == p_filt[1]
        t += dt


def test_one_euro_filter_2d_jitter_attenuation_at_30fps():
    # SLA: Fixation Stability (Jitter) < 5 px RMS a 30 FPS
    f = OneEuroFilter2D(min_cutoff=0.8, beta=0.004, d_cutoff=1.0, isotropic=True)

    t = 0.0
    dt = 0.0333  # 30 FPS
    np.random.seed(42)
    # Sinal estacionário em (960, 540) com ruído gaussiano sigma = 3.0 px (6 px peak-to-peak)
    noise_x = np.random.normal(0, 3.0, 60)
    noise_y = np.random.normal(0, 3.0, 60)

    filtered_pts = []
    for i in range(60):
        p_raw = np.array([960.0 + noise_x[i], 540.0 + noise_y[i]])
        p_filt = f(p_raw, t)
        filtered_pts.append(p_filt)
        t += dt

    filtered_arr = np.array(filtered_pts)[20:]  # descarta regime transitório
    rms_jitter_x = float(np.std(filtered_arr[:, 0]))
    rms_jitter_y = float(np.std(filtered_arr[:, 1]))
    rms_total = float(np.hypot(rms_jitter_x, rms_jitter_y))

    # Jitter bruto RMS é ~4.2 px. Filtrado deve ser < 2.5 px RMS (bem abaixo do SLA de 5 px)
    assert rms_total < 2.5


def test_fixation_stabilizer_smooth_continuous_anchor():
    # Valida eliminação do salto descontínuo de 45px (leaky anchor contínua C^0)
    stabilizer = FixationStabilizer(radius_px=45.0, smooth=True, break_mult=2.5)

    p0 = np.array([500.0, 500.0])
    stabilizer(p0, 0.0)

    # 1. Ponto logo além da borda do raio (dist = 46 px, apenas 1 px de excesso)
    p_edge = np.array([546.0, 500.0])
    res_edge = stabilizer(p_edge, 0.033)

    # No modo clássico descontínuo pularia 46px direto para (546, 500).
    # Com âncora contínua, move suavemente apenas o excesso de 1px: centro vai para (501, 500)!
    assert pytest.approx(res_edge[0], abs=1e-3) == 501.0
    assert pytest.approx(res_edge[1], abs=1e-3) == 500.0

    # 2. Deslocamento contínuo em 30 FPS: avança 5 px por quadro
    p_next = np.array([551.0, 500.0])
    res_next = stabilizer(p_next, 0.066)
    assert pytest.approx(res_next[0], abs=1e-3) == 506.0


def test_fixation_stabilizer_saccadic_break():
    # Salto amplo (> break_mult * radius = 2.5 * 45 = 112.5 px) rompe a âncora imediatamente
    stabilizer = FixationStabilizer(radius_px=45.0, smooth=True, break_mult=2.5)

    p0 = np.array([500.0, 500.0])
    stabilizer(p0, 0.0)

    # Sacada de 200px
    p_saccade = np.array([700.0, 500.0])
    res_saccade = stabilizer(p_saccade, 0.033)

    # Rompe e ancora diretamente no novo ponto
    assert np.allclose(res_saccade, [700.0, 500.0])
    assert stabilizer.fixation_start == pytest.approx(0.033)


def test_one_euro_filter_2d_isotropic_vs_separable_speed():
    # Quando o movimento é oblíquo ou assimétrico (dx >> dy),
    # o filtro separável abre apenas em X e atrasa em Y.
    # O filtro isotrópico calcula a velocidade vetorial conjunta 2D,
    # expandindo o corte em ambos os eixos e prevenindo distorção angular.
    f_sep = OneEuroFilter2D(min_cutoff=0.8, beta=0.01, d_cutoff=1.0, isotropic=False)
    f_iso = OneEuroFilter2D(min_cutoff=0.8, beta=0.01, d_cutoff=1.0, isotropic=True)

    t = 0.0
    dt = 0.0333
    f_sep(np.array([100.0, 100.0]), t)
    f_iso(np.array([100.0, 100.0]), t)

    t += dt
    raw_step = np.array([300.0, 110.0])
    res_sep = f_sep(raw_step, t)
    res_iso = f_iso(raw_step, t)
    assert res_iso[1] > res_sep[1]


def test_position_history_immutable_snapshots():
    # Valida que modificações in-place subsequentes em arrays não corrompem o histórico passado
    hist = PositionHistory(maxlen=10)
    arr = np.array([100.0, 100.0])
    hist.push(1.0, arr)

    # Modifica in-place o array original
    arr[0] = 999.0
    arr[1] = 888.0

    p_saved = hist.at(1.0)
    assert p_saved is not None
    assert p_saved[0] == pytest.approx(100.0)
    assert p_saved[1] == pytest.approx(100.0)


def test_fixation_stabilizer_smooth_pursuit_settles_to_zero_offset():
    # Valida que após perseguição suave o cursor converge assintoticamente ao alvo (offset zero)
    # eliminando a histerese permanente de 45px do arrasto
    stabilizer = FixationStabilizer(radius_px=45.0, smooth=True, break_mult=2.5, catchup_rate=0.25)

    # Inicia em 100.0
    stabilizer(np.array([100.0, 100.0]), 0.0)

    # Move continuamente em passos de 20px até 500.0
    t = 0.033
    for i in range(20):
        p = np.array([100.0 + (i + 1) * 20.0, 100.0])
        stabilizer(p, t)
        t += 0.033

    # No final do movimento, o olhar para em 500.0
    # O estabilizador deve convergir suavemente até zerar a diferença (< 1.0 px)
    for _ in range(25):
        res = stabilizer(np.array([500.0, 100.0]), t)
        t += 0.033

    assert pytest.approx(res[0], abs=1.0) == 500.0
    assert pytest.approx(res[1], abs=1.0) == 100.0
    assert not stabilizer._dragging
