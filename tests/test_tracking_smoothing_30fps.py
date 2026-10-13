import numpy as np
import pytest

from lorena.config import Settings
from lorena.domain.models import GazeFeatures, GestureEvent, TrackingState
from lorena.mapping.gaze_mapper import GazeMapper
from lorena.services.cursor_session import CursorSession


def test_blink_freeze_debounce_ignores_1frame_spike(mock_cursor):
    """Valida que ruído de 1 único quadro (flicker de pálpebra a 30 FPS) não congela o cursor."""
    mapper = GazeMapper(degree=1, auto_select=False, screen_size=(1920, 1080))
    mapper._mu = np.array([0.5, 0.0])
    mapper._sigma = np.array([0.1, 0.1])
    mapper._beta = np.array([[960.0, 540.0], [500.0, 0.0], [0.0, 300.0]])
    mapper._zmin = np.array([-5.0, -5.0])
    mapper._zmax = np.array([5.0, 5.0])

    settings = Settings()
    settings.filters.freeze_confirm_frames = 2
    session = CursorSession(mapper, mock_cursor, settings)

    t = 1.000
    f_open = GazeFeatures(
        timestamp=t, h=0.5, v=0.0, yaw=0.0, pitch=0.0,
        ear_left=0.3, ear_right=0.3, openness_left=1.0, openness_right=1.0, quality=1.0
    )
    session.on_frame(f_open, t)
    assert np.allclose(mock_cursor.pos, [960.0, 540.0])

    # 1. Spike de 1 quadro com olho esquerdo fechado (< 0.30):
    t += 0.033
    f_spike = GazeFeatures(
        timestamp=t, h=0.51, v=0.0, yaw=0.0, pitch=0.0,
        ear_left=0.1, ear_right=0.3, openness_left=0.15, openness_right=0.90, quality=1.0
    )
    session.on_frame(f_spike, t)
    # Com debounce de 2 quadros, o cursor NÃO deve congelar no primeiro quadro isolado
    assert session._blink_freeze_frames == 1
    assert session._frozen_at is None
    # Cursor continua se movendo sem travamento perceptível
    assert mock_cursor.pos[0] >= 960.0

    # 2. Próximo quadro volta ao normal:
    t += 0.033
    f_normal = GazeFeatures(
        timestamp=t, h=0.52, v=0.0, yaw=0.0, pitch=0.0,
        ear_left=0.3, ear_right=0.3, openness_left=1.0, openness_right=1.0, quality=1.0
    )
    session.on_frame(f_normal, t)
    assert session._blink_freeze_frames == 0
    assert session._frozen_at is None


def test_blink_freeze_confirms_sustained_wink(mock_cursor):
    """Valida que gesto intencional sustentado por >= 2 quadros ativa o congelamento pré-clique."""
    mapper = GazeMapper(degree=1, auto_select=False, screen_size=(1920, 1080))
    mapper._mu = np.array([0.5, 0.0])
    mapper._sigma = np.array([0.1, 0.1])
    mapper._beta = np.array([[960.0, 540.0], [500.0, 0.0], [0.0, 300.0]])
    mapper._zmin = np.array([-5.0, -5.0])
    mapper._zmax = np.array([5.0, 5.0])

    settings = Settings()
    settings.filters.freeze_confirm_frames = 2
    session = CursorSession(mapper, mock_cursor, settings)

    t = 1.000
    for _ in range(5):
        f = GazeFeatures(
            timestamp=t, h=0.5, v=0.0, yaw=0.0, pitch=0.0,
            ear_left=0.3, ear_right=0.3, openness_left=1.0, openness_right=1.0, quality=1.0
        )
        session.on_frame(f, t)
        t += 0.033

    # Quadro 1 do wink:
    f_wink1 = GazeFeatures(
        timestamp=t, h=0.5, v=0.0, yaw=0.0, pitch=0.0,
        ear_left=0.1, ear_right=0.3, openness_left=0.15, openness_right=0.90, quality=1.0
    )
    session.on_frame(f_wink1, t)
    assert session._blink_freeze_frames == 1

    # Quadro 2 do wink (confirmado):
    t += 0.033
    f_wink2 = GazeFeatures(
        timestamp=t, h=0.5, v=0.0, yaw=0.0, pitch=0.0,
        ear_left=0.1, ear_right=0.3, openness_left=0.15, openness_right=0.90, quality=1.0
    )
    session.on_frame(f_wink2, t)
    assert session._blink_freeze_frames == 2
    # Agora deve congelar
    assert session._frozen_at is not None


def test_smooth_pursuit_continuous_tracking_30fps(mock_cursor):
    """Valida perseguição suave contínua a 30 FPS sem efeito escada/catraca de 45px."""
    mapper = GazeMapper(degree=1, auto_select=False, screen_size=(1920, 1080))
    mapper._mu = np.array([0.5, 0.0])
    mapper._sigma = np.array([0.1, 0.1])
    mapper._beta = np.array([[960.0, 540.0], [500.0, 0.0], [0.0, 300.0]])
    mapper._zmin = np.array([-5.0, -5.0])
    mapper._zmax = np.array([5.0, 5.0])

    settings = Settings()
    settings.filters.smooth_deadzone = True
    session = CursorSession(mapper, mock_cursor, settings)

    t = 1.000
    positions = []
    # Simula 30 quadros com olhar deslizando lentamente para a direita
    for i in range(30):
        h = 0.5 + (i * 0.005)  # avanço suave de ~25px a cada frame
        f = GazeFeatures(
            timestamp=t, h=h, v=0.0, yaw=0.0, pitch=0.0,
            ear_left=0.3, ear_right=0.3, openness_left=1.0, openness_right=1.0, quality=1.0
        )
        session.on_frame(f, t)
        positions.append(mock_cursor.pos.copy())
        t += 0.0333

    # As diferenças entre passos consecutivos não devem ter saltos de 45px
    diffs = [positions[i][0] - positions[i - 1][0] for i in range(1, len(positions))]
    max_step = max(diffs[5:])  # após acomodação inicial
    assert max_step < 35.0, f"Salto máximo {max_step} px excede suavização contínua"


def test_smooth_pursuit_and_steady_fixation_zero_offset(mock_cursor):
    """Valida que após perseguição o cursor assenta exatamente no alvo (< 2 px offset).
    
    Elimina a falha anterior onde o cursor parava permanentemente a 40-45 px do alvo.
    """
    mapper = GazeMapper(degree=1, auto_select=False, screen_size=(1920, 1080))
    mapper._mu = np.array([0.5, 0.0])
    mapper._sigma = np.array([0.1, 0.1])
    mapper._beta = np.array([[960.0, 540.0], [500.0, 0.0], [0.0, 300.0]])
    mapper._zmin = np.array([-5.0, -5.0])
    mapper._zmax = np.array([5.0, 5.0])

    settings = Settings()
    settings.filters.smooth_deadzone = True
    session = CursorSession(mapper, mock_cursor, settings)

    t = 1.000
    # 1. Perseguição suave avançando h de 0.5 para 0.645 (~1685 px na tela)
    for i in range(30):
        h = 0.5 + (i * 0.005)
        f = GazeFeatures(
            timestamp=t, h=h, v=0.0, yaw=0.0, pitch=0.0,
            ear_left=0.3, ear_right=0.3, openness_left=1.0, openness_right=1.0, quality=1.0
        )
        session.on_frame(f, t)
        t += 0.0333

    target_gaze_x = mapper.predict(f)[0]

    # 2. Fixação estável de 1 segundo (30 quadros) exatamente no alvo final
    for _ in range(30):
        f = GazeFeatures(
            timestamp=t, h=0.645, v=0.0, yaw=0.0, pitch=0.0,
            ear_left=0.3, ear_right=0.3, openness_left=1.0, openness_right=1.0, quality=1.0
        )
        session.on_frame(f, t)
        t += 0.0333

    # O offset residual deve ser desprezível (< 2.0 px, superando SLA de < 30 px)
    offset = abs(mock_cursor.pos[0] - target_gaze_x)
    assert offset < 2.0, f"Offset final {offset} px excede limite aceitável de 2.0 px"


def test_blink_freeze_low_fps_resilience(mock_cursor):
    """Valida resiliência de gestos e rollback sob quedas severas de FPS (10 FPS / dt=100ms)."""
    mapper = GazeMapper(degree=1, auto_select=False, screen_size=(1920, 1080))
    mapper._mu = np.array([0.5, 0.0])
    mapper._sigma = np.array([0.1, 0.1])
    mapper._beta = np.array([[960.0, 540.0], [500.0, 0.0], [0.0, 300.0]])
    mapper._zmin = np.array([-5.0, -5.0])
    mapper._zmax = np.array([5.0, 5.0])

    settings = Settings()
    settings.filters.freeze_confirm_frames = 2
    settings.blink.min_wink_s = 0.20
    session = CursorSession(mapper, mock_cursor, settings)

    t = 1.000
    dt_slow = 0.100  # 10 FPS

    # 5 quadros nominais a 10 FPS
    for _ in range(5):
        f = GazeFeatures(
            timestamp=t, h=0.5, v=0.0, yaw=0.0, pitch=0.0,
            ear_left=0.3, ear_right=0.3, openness_left=1.0, openness_right=1.0, quality=1.0
        )
        session.on_frame(f, t)
        t += dt_slow

    # Inicia wink: Quadro 1 fechado (t = 1.500)
    f_wink1 = GazeFeatures(
        timestamp=t, h=0.5, v=0.0, yaw=0.0, pitch=0.0,
        ear_left=0.1, ear_right=0.3, openness_left=0.15, openness_right=0.90, quality=1.0
    )
    session.on_frame(f_wink1, t)

    # Quadro 2 fechado: t = 1.600 (+100ms)
    t += dt_slow
    f_wink2 = GazeFeatures(
        timestamp=t, h=0.5, v=0.0, yaw=0.0, pitch=0.0,
        ear_left=0.1, ear_right=0.3, openness_left=0.15, openness_right=0.90, quality=1.0
    )
    session.on_frame(f_wink2, t)

    # Quadro 3 reabre: t = 1.700 (+100ms, duração = 0.200s >= min_wink_s)
    t += dt_slow
    f_open = GazeFeatures(
        timestamp=t, h=0.5, v=0.0, yaw=0.0, pitch=0.0,
        ear_left=0.3, ear_right=0.3, openness_left=1.0, openness_right=1.0, quality=1.0
    )
    status = session.on_frame(f_open, t)

    # Deve registrar clique com sucesso mesmo a 10 FPS
    assert status.last_event is GestureEvent.CLICK
    assert "left" in mock_cursor.clicks


def test_candidate_freeze_frame_does_not_move_cursor_to_glitch(mock_cursor):
    """Valida que o quadro 1 de queda de EAR não desloca o cursor para a íris distorcida pela pálpebra."""
    mapper = GazeMapper(degree=1, auto_select=False, screen_size=(1920, 1080))
    mapper._mu = np.array([0.5, 0.0])
    mapper._sigma = np.array([0.1, 0.1])
    mapper._beta = np.array([[960.0, 540.0], [500.0, 0.0], [0.0, 300.0]])
    mapper._zmin = np.array([-5.0, -5.0])
    mapper._zmax = np.array([5.0, 5.0])

    settings = Settings()
    settings.filters.freeze_confirm_frames = 2
    session = CursorSession(mapper, mock_cursor, settings)

    t = 1.000
    f_stable = GazeFeatures(
        timestamp=t, h=0.5, v=0.0, yaw=0.0, pitch=0.0,
        ear_left=0.3, ear_right=0.3, openness_left=1.0, openness_right=1.0, quality=1.0
    )
    session.on_frame(f_stable, t)
    p_initial = mock_cursor.pos.copy()

    # Quadro com fechamento parcial onde h salta 50px de ruído (h=0.51)
    t += 0.033
    f_corrupted = GazeFeatures(
        timestamp=t, h=0.51, v=0.0, yaw=0.0, pitch=0.0,
        ear_left=0.1, ear_right=0.3, openness_left=0.15, openness_right=0.90, quality=1.0
    )
    session.on_frame(f_corrupted, t)

    # O cursor NÃO deve se mover para o ruído
    assert np.allclose(mock_cursor.pos, p_initial)

