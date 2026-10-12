"""Mapeamento (h, v[, yaw, pitch]) → pixel por regressão polinomial com ridge.

Substitui o antigo mapeamento linear separável (que usava só as bordas da
grade, ignorava o ponto central e o deslocamento dos pontos em relação à
borda da tela):

    φ(h, v) = [1, h, v, h·v, h², v²] (+ yaw, pitch)
    β = (ΦᵀΦ + λI)⁻¹ Φᵀ T        (intercepto não penalizado)

Features são normalizadas (z-score) antes do ajuste para condicionar o sistema.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lorena.domain.models import CalibrationSample, GazeFeatures


class NotFittedError(RuntimeError):
    pass


@dataclass
class CalibrationReport:
    mean_error_px: float
    loo_mean_error_px: float | None
    per_point_error_px: dict[int, float] = field(default_factory=dict)
    num_samples: int = 0
    num_points: int = 0


class GazeMapper:
    VERSION = 1

    def __init__(self, degree: int = 2, ridge_lambda: float = 1e-3, use_head_pose: bool = False,
                 screen_size: tuple[int, int] = (1920, 1080), offset: tuple[float, float] = (0.0, 0.0)) -> None:
        if degree not in (1, 2):
            raise ValueError("degree deve ser 1 ou 2")
        self.degree = degree
        self.ridge_lambda = ridge_lambda
        self.use_head_pose = use_head_pose
        self.screen_size = screen_size
        self.offset = np.array(offset, dtype=float)
        self._mu: np.ndarray | None = None
        self._sigma: np.ndarray | None = None
        self._beta: np.ndarray | None = None

    # ------------------------------------------------------------------ design
    def _raw(self, h: np.ndarray, v: np.ndarray, yaw: np.ndarray, pitch: np.ndarray) -> np.ndarray:
        cols = [h, v]
        if self.use_head_pose:
            cols += [yaw, pitch]
        return np.column_stack(cols)

    def _design(self, raw: np.ndarray) -> np.ndarray:
        z = (raw - self._mu) / self._sigma
        h, v = z[:, 0], z[:, 1]
        cols = [np.ones(len(z)), h, v]
        if self.degree == 2:
            cols += [h * v, h ** 2, v ** 2]
        if self.use_head_pose:
            cols += [z[:, 2], z[:, 3]]
        return np.column_stack(cols)

    @property
    def num_terms(self) -> int:
        return (3 if self.degree == 1 else 6) + (2 if self.use_head_pose else 0)

    @property
    def is_fitted(self) -> bool:
        return self._beta is not None

    def set_drift_offset(self, offset: tuple[float, float]) -> None:
        """Define o deslocamento estático de correção de deriva."""
        self.offset = np.array(offset, dtype=float)

    def reset_drift_offset(self) -> None:
        """Zera qualquer compensação de deriva ativa."""
        self.offset = np.zeros(2, dtype=float)

    def recalibrate_drift(self, f: GazeFeatures, target: tuple[float, float]) -> tuple[float, float]:
        """Calcula e aplica offset de 1 ponto contra o alvo especificado."""
        base_pred = self.predict_unbiased(f)
        dx = float(target[0] - base_pred[0])
        dy = float(target[1] - base_pred[1])
        self.set_drift_offset((dx, dy))
        return (dx, dy)

    # --------------------------------------------------------------------- fit
    @staticmethod
    def _arrays(samples: list[CalibrationSample]):
        a = np.array([(s.h, s.v, s.yaw, s.pitch, s.target_x, s.target_y, s.point_index) for s in samples], dtype=float)
        return a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4:6], a[:, 6].astype(int)

    def _solve(self, raw: np.ndarray, targets: np.ndarray) -> None:
        self._mu = raw.mean(axis=0)
        self._sigma = raw.std(axis=0)
        self._sigma[self._sigma < 1e-9] = 1.0
        x = self._design(raw)
        penalty = self.ridge_lambda * np.eye(x.shape[1])
        penalty[0, 0] = 0.0
        self._beta = np.linalg.solve(x.T @ x + penalty, x.T @ targets)

    def fit(self, samples: list[CalibrationSample]) -> CalibrationReport:
        h, v, yaw, pitch, targets, idx = self._arrays(samples)
        points = np.unique(idx)
        min_required = self.num_terms - (2 if self.use_head_pose else 0)
        if len(points) < min_required:
            raise ValueError(f"Pontos insuficientes ({len(points)}) para grau {self.degree}.")
        if self.use_head_pose and len(points) < self.num_terms:
            self.use_head_pose = False
        raw = self._raw(h, v, yaw, pitch)

        loo_errors = []
        if len(points) > self.num_terms:
            for p in points:
                train = idx != p
                self._solve(raw[train], targets[train])
                pred = self._design(raw[~train]) @ self._beta
                loo_errors.append(np.linalg.norm(pred.mean(axis=0) - targets[~train][0]))

        self._solve(raw, targets)
        pred = self._design(raw) @ self._beta
        per_point = {int(p): float(np.linalg.norm(pred[idx == p].mean(axis=0) - targets[idx == p][0])) for p in points}
        return CalibrationReport(
            mean_error_px=float(np.mean(list(per_point.values()))),
            loo_mean_error_px=float(np.mean(loo_errors)) if loo_errors else None,
            per_point_error_px=per_point,
            num_samples=len(samples),
            num_points=len(points),
        )

    # ----------------------------------------------------------------- predict
    def predict_unbiased(self, f: GazeFeatures) -> np.ndarray:
        return self.predict_raw(f.h, f.v, f.yaw, f.pitch, apply_offset=False)

    def predict(self, f: GazeFeatures) -> np.ndarray:
        return self.predict_raw(f.h, f.v, f.yaw, f.pitch, apply_offset=True)

    def predict_raw(self, h: float, v: float, yaw: float = 0.0, pitch: float = 0.0, apply_offset: bool = True) -> np.ndarray:
        if self._beta is None:
            raise NotFittedError("GazeMapper não calibrado.")
        raw = self._raw(np.array([h]), np.array([v]), np.array([yaw]), np.array([pitch]))
        p = (self._design(raw) @ self._beta)[0]
        if apply_offset:
            p = p + self.offset
        w, hgt = self.screen_size
        return np.array([np.clip(p[0], 0, w - 1), np.clip(p[1], 0, hgt - 1)])

    # ----------------------------------------------------------- serialization
    def to_dict(self) -> dict:
        if self._beta is None:
            raise NotFittedError("GazeMapper não calibrado.")
        return {
            "version": self.VERSION,
            "degree": self.degree,
            "ridge_lambda": self.ridge_lambda,
            "use_head_pose": self.use_head_pose,
            "screen_size": list(self.screen_size),
            "offset": self.offset.tolist(),
            "mu": np.asarray(self._mu).tolist(),
            "sigma": np.asarray(self._sigma).tolist(),
            "beta": np.asarray(self._beta).tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GazeMapper":
        if d.get("version") != cls.VERSION:
            raise ValueError("Versão de mapeador incompatível.")
        offset = tuple(d.get("offset", [0.0, 0.0]))
        m = cls(d["degree"], d["ridge_lambda"], d["use_head_pose"], tuple(d["screen_size"]), offset=offset)
        m._mu = np.array(d["mu"], dtype=float)
        m._sigma = np.array(d["sigma"], dtype=float)
        m._beta = np.array(d["beta"], dtype=float)
        if m._beta.shape != (m.num_terms, 2):
            raise ValueError("Coeficientes com formato inválido.")
        return m
