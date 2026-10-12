import json
import pytest

from lorena.infra.calibration_repo import CalibrationProfile, CalibrationRepository
from lorena.mapping.gaze_mapper import CalibrationReport, GazeMapper


def test_repo_save_and_load(tmp_path):
    repo = CalibrationRepository(tmp_path)

    mapper = GazeMapper(degree=1, screen_size=(1920, 1080))
    # Synthetic fit
    mapper._mu = [0.5, 0.0]
    mapper._sigma = [0.1, 0.1]
    mapper._beta = [[960.0, 540.0], [500.0, 0.0], [0.0, 300.0]]

    report = CalibrationReport(
        mean_error_px=15.2,
        loo_mean_error_px=18.4,
        per_point_error_px={0: 12.0, 1: 18.4},
        num_samples=30,
        num_points=9
    )

    profile = CalibrationProfile(
        mapper=mapper,
        open_ear_left=0.32,
        open_ear_right=0.31,
        screen_size=(1920, 1080),
        report=report
    )

    saved_path = repo.save("test_profile", profile)
    assert saved_path.exists()

    loaded = repo.load("test_profile")
    assert loaded is not None
    assert loaded.screen_size == (1920, 1080)
    assert pytest.approx(loaded.open_ear_left, abs=1e-4) == 0.32
    assert pytest.approx(loaded.report.mean_error_px, abs=1e-2) == 15.2
    assert loaded.report.per_point_error_px[0] == 12.0


def test_repo_load_nonexistent_or_corrupt(tmp_path):
    repo = CalibrationRepository(tmp_path)
    assert repo.load("missing") is None

    # Write invalid json
    corrupt_file = tmp_path / "profile_corrupt.json"
    corrupt_file.write_text("{invalid json", encoding="utf-8")
    assert repo.load("corrupt") is None
