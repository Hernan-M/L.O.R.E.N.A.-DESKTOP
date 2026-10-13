"""Avaliador quantitativo de acurácia de rastreio ocular (em pixels e graus de campo visual)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from lorena.domain.models import GazeFeatures
from lorena.infra.platform import screen_pixel_pitch_mm
from lorena.mapping.gaze_mapper import GazeMapper


@dataclass(frozen=True)
class TargetAccuracy:
    target_x: float
    target_y: float
    error_px: float
    error_deg: float
    region: str = "center"  # "center", "edge", "corner"


@dataclass(frozen=True)
class AccuracyReport:
    mean_error_px: float
    mean_error_deg: float
    max_error_px: float
    max_error_deg: float
    samples_count: int
    targets: list[TargetAccuracy] = field(default_factory=list)
    grade: str = "Bom"  # "Excelente", "Bom", "Aceitável", "Necessita Recalibração"
    region_error_px: dict[str, float] = field(default_factory=dict)
    region_error_deg: dict[str, float] = field(default_factory=dict)


class AccuracyEvaluator:
    """Calcula métricas quantitativas de acurácia em pixels e graus visuais."""

    def __init__(
        self,
        mapper: GazeMapper,
        screen_size: tuple[int, int] = (1920, 1080),
        viewing_distance_mm: float = 600.0,
        pixel_pitch_mm: float | None = None,
    ) -> None:
        self.mapper = mapper
        self.screen_size = screen_size
        self.viewing_distance_mm = viewing_distance_mm
        self.pixel_pitch_mm = pixel_pitch_mm if pixel_pitch_mm is not None else screen_pixel_pitch_mm()

    def classify_region(self, x: float, y: float) -> str:
        """Classifica se o ponto está no centro, borda ou canto da tela."""
        w, h = self.screen_size
        frac_x = min(x, w - x) / max(w, 1)
        frac_y = min(y, h - y) / max(h, 1)
        if frac_x <= 0.20 and frac_y <= 0.20:
            return "corner"
        if frac_x <= 0.15 or frac_y <= 0.15:
            return "edge"
        return "center"

    def pixels_to_degrees(self, error_px: float) -> float:
        """Converte erro euclidiano em pixels para graus de ângulo visual."""
        error_mm = error_px * self.pixel_pitch_mm
        return float(math.degrees(math.atan2(error_mm, self.viewing_distance_mm)))

    def evaluate_points(
        self,
        evaluations: Sequence[tuple[tuple[float, float], Sequence[GazeFeatures]]],
    ) -> AccuracyReport:
        """Avalia previsões contra uma sequência de alvos e suas respectivas amostras de olhar."""
        target_results: list[TargetAccuracy] = []

        for target, samples in evaluations:
            if not samples:
                continue
            preds = [self.mapper.predict(f) for f in samples]
            mean_pred = np.mean(preds, axis=0)
            err_px = float(np.linalg.norm(mean_pred - np.array(target)))
            err_deg = self.pixels_to_degrees(err_px)
            region = self.classify_region(float(target[0]), float(target[1]))
            target_results.append(
                TargetAccuracy(
                    target_x=float(target[0]),
                    target_y=float(target[1]),
                    error_px=err_px,
                    error_deg=err_deg,
                    region=region,
                )
            )

        if not target_results:
            return AccuracyReport(0.0, 0.0, 0.0, 0.0, 0, [], "Sem dados")

        mean_px = float(np.mean([t.error_px for t in target_results]))
        mean_deg = float(np.mean([t.error_deg for t in target_results]))
        max_px = float(np.max([t.error_px for t in target_results]))
        max_deg = float(np.max([t.error_deg for t in target_results]))
        total_samples = sum(len(samples) for _, samples in evaluations)

        # Agrupamento por região (centro / borda / canto)
        reg_px: dict[str, float] = {}
        reg_deg: dict[str, float] = {}
        for r_name in ("center", "edge", "corner"):
            sub = [t for t in target_results if t.region == r_name]
            if sub:
                reg_px[r_name] = float(np.mean([t.error_px for t in sub]))
                reg_deg[r_name] = float(np.mean([t.error_deg for t in sub]))

        # Classificação clínica / ergonômica padrão de Eye Tracking
        if mean_deg < 1.5:
            grade = "Excelente"
        elif mean_deg < 2.5:
            grade = "Bom"
        elif mean_deg < 4.0:
            grade = "Aceitável"
        else:
            grade = "Necessita Recalibração"

        return AccuracyReport(
            mean_error_px=mean_px,
            mean_error_deg=mean_deg,
            max_error_px=max_px,
            max_error_deg=max_deg,
            samples_count=total_samples,
            targets=target_results,
            grade=grade,
            region_error_px=reg_px,
            region_error_deg=reg_deg,
        )

    def generate_random_targets(self, count: int = 5, margin_px: int = 100) -> list[tuple[float, float]]:
        """Gera pontos de teste aleatórios bem distribuídos na tela."""
        w, h = self.screen_size
        rng = np.random.default_rng()
        xs = rng.uniform(margin_px, w - margin_px, size=count)
        ys = rng.uniform(margin_px, h - margin_px, size=count)
        return [(float(x), float(y)) for x, y in zip(xs, ys)]
