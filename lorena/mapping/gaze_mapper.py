"""Mapeamento (h, v[, lid][, yaw, pitch, hx, hy]) → pixel por regressão polinomial com ridge.

    φ(h, v) = [1, h, v, h·v, h², v²]  (+ lid)  (+ yaw, pitch, hx, hy)
    β = (ΦᵀΦ + Λ)⁻¹ Φᵀ T             (intercepto não penalizado)

Melhorias da v0.3.1 (ver docs/PLANO_DE_MELHORIAS.md, B.6):

* **Seleção de modelo por Leave-One-Out** (``auto_select``): testa grau 1/2, com/sem
  ``lid``, com/sem cabeça e vários λ; só entram candidatos com
  ``n_pontos >= n_termos + 3`` (antes: 8 termos para 9 pontos = interpolação exata e
  erro LOO ≈ 5× o de treino).
* **Termos de cabeça com escala física fixa** (0,1 rad / 0,1 adimensional) e ridge
  mais forte; o coeficiente é limitado a um valor fisicamente plausível
  (``head_limit_px``), então ruído de pose não vira "voo" do cursor.
* **Extrapolação limitada**: h/v são limitados à faixa calibrada (+15 %) antes do
  polinômio, evitando que o termo quadrático divirja nos cantos.
* **Drift com offset + escala** (≥ 2 pontos de recalibração).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lorena.domain.models import CalibrationSample, GazeFeatures

# Escalas fixas (z = (x - ref) / escala) para colunas de cabeça: independem da
# variação (possivelmente minúscula) observada durante a calibração.
POSE_SCALES = np.array([0.1, 0.1, 0.1, 0.1])  # yaw, pitch (rad), hx, hy (adimensional)
LAMBDA_GRID = (1e-3, 1e-2, 1e-1, 1.0)
MIN_POSE_STD = 0.012  # abaixo disso (≈0,7°) a pose não é identificável: modelo sem cabeça
EXTRAP_MARGIN = 0.15


class NotFittedError(RuntimeError):
    pass


@dataclass
class CalibrationReport:
    mean_error_px: float
    loo_mean_error_px: float | None
    per_point_error_px: dict[int, float] = field(default_factory=dict)
    num_samples: int = 0
    num_points: int = 0
    model: dict = field(default_factory=dict)
    face_luma: float | None = None


class GazeMapper:
    VERSION = 2

    def __init__(self, degree: int = 2, ridge_lambda: float = 1e-3, use_head_pose: bool = False,
                 screen_size: tuple[int, int] = (1920, 1080), offset: tuple[float, float] = (0.0, 0.0),
                 use_lid: bool = False, auto_select: bool = False, pose_ridge_factor: float = 10.0,
                 head_limit_px: float = 450.0, basis: str = "symmetric") -> None:
        if degree not in (1, 2):
            raise ValueError("degree deve ser 1 ou 2")
        self.degree = degree
        self.ridge_lambda = ridge_lambda
        self.use_head_pose = use_head_pose
        self.use_lid = use_lid
        self.auto_select = auto_select
        self.pose_ridge_factor = pose_ridge_factor
        self.head_limit_px = head_limit_px
        self.basis = basis
        self.screen_size = screen_size
        self.offset = np.array(offset, dtype=float)
        self.scale = np.ones(2, dtype=float)
        self._mu: np.ndarray | None = None
        self._sigma: np.ndarray | None = None
        self._beta: np.ndarray | None = None
        self._zmin: np.ndarray | None = None  # faixa calibrada de (h, v) em z
        self._zmax: np.ndarray | None = None
        self._pose_max = np.zeros(2)          # |Δyaw|, |Δpitch| máximos vistos na calibração

    # ------------------------------------------------------------------ layout
    @property
    def _n_extra(self) -> int:
        return (1 if self.use_lid else 0) + (4 if self.use_head_pose else 0)

    @property
    def _pose_col(self) -> int:
        return 2 + (1 if self.use_lid else 0)

    def _raw(self, h, v, yaw, pitch, lid=None, hx=None, hy=None) -> np.ndarray:
        h = np.atleast_1d(np.asarray(h, dtype=float))
        zeros = np.zeros_like(h)
        cols = [h, np.atleast_1d(np.asarray(v, dtype=float))]
        if self.use_lid:
            cols.append(zeros if lid is None else np.atleast_1d(np.asarray(lid, dtype=float)))
        if self.use_head_pose:
            cols += [np.atleast_1d(np.asarray(yaw, dtype=float)), np.atleast_1d(np.asarray(pitch, dtype=float)),
                     zeros if hx is None else np.atleast_1d(np.asarray(hx, dtype=float)),
                     zeros if hy is None else np.atleast_1d(np.asarray(hy, dtype=float))]
        return np.column_stack(cols)

    def _design(self, raw: np.ndarray, clip: bool = False) -> np.ndarray:
        z = (raw - self._mu) / self._sigma
        if clip and self._zmin is not None:
            z = z.copy()
            span = self._zmax - self._zmin
            z[:, :2] = np.clip(z[:, :2], self._zmin - EXTRAP_MARGIN * span, self._zmax + EXTRAP_MARGIN * span)
        h, v = z[:, 0], z[:, 1]
        cols = [np.ones(len(z)), h, v]
        if self.degree == 2:
            if getattr(self, "basis", "symmetric") == "symmetric":
                cols += [h * v, h * np.abs(h), v * np.abs(v)]
            else:
                cols += [h * v, h ** 2, v ** 2]
        for j in range(2, z.shape[1]):
            cols.append(z[:, j])
        return np.column_stack(cols)

    @property
    def num_terms(self) -> int:
        return (3 if self.degree == 1 else 6) + self._n_extra

    @property
    def is_fitted(self) -> bool:
        return self._beta is not None

    # ------------------------------------------------------------------- drift
    def set_drift_offset(self, offset: tuple[float, float]) -> None:
        """Define o deslocamento estático de correção de deriva."""
        self.offset = np.array(offset, dtype=float)

    def reset_drift_offset(self) -> None:
        """Zera qualquer compensação de deriva ativa (offset e escala)."""
        self.offset = np.zeros(2, dtype=float)
        self.scale = np.ones(2, dtype=float)

    def recalibrate_drift(self, f: GazeFeatures, target: tuple[float, float]) -> tuple[float, float]:
        """Calcula e aplica offset de 1 ponto contra o alvo especificado."""
        base_pred = self.predict_unbiased(f)
        dx = float(target[0] - base_pred[0])
        dy = float(target[1] - base_pred[1])
        self.scale = np.ones(2, dtype=float)
        self.set_drift_offset((dx, dy))
        return (dx, dy)

    def recalibrate_drift_multi(self, pairs: list[tuple[GazeFeatures, tuple[float, float]]]) -> tuple[np.ndarray, np.ndarray]:
        """Ajusta offset + escala (em torno do centro da tela) por mínimos quadrados."""
        if len(pairs) < 2:
            raise ValueError("São necessários ao menos 2 pontos.")
        center = np.array(self.screen_size, dtype=float) / 2.0
        pred = np.array([self.predict_unbiased(f) for f, _ in pairs]) - center
        tgt = np.array([t for _, t in pairs], dtype=float) - center
        scale = np.ones(2)
        offset = np.zeros(2)
        for a in range(2):
            x = np.column_stack([pred[:, a], np.ones(len(pairs))])
            if np.ptp(pred[:, a]) < 1.0:  # sem amplitude: só offset
                offset[a] = float(np.mean(tgt[:, a] - pred[:, a]))
                continue
            s, o = np.linalg.lstsq(x, tgt[:, a], rcond=None)[0]
            scale[a], offset[a] = float(np.clip(s, 0.7, 1.4)), float(o)
        self.scale, self.offset = scale, offset
        return scale, offset

    # --------------------------------------------------------------------- fit
    @staticmethod
    def _arrays(samples: list[CalibrationSample]):
        a = np.array([(s.h, s.v, s.yaw, s.pitch, s.target_x, s.target_y, s.point_index,
                       s.lid, s.head_x, s.head_y) for s in samples], dtype=float)
        return a

    def _penalty(self, lam: float, n_cols: int) -> np.ndarray:
        diag = np.full(n_cols, lam)
        diag[0] = 0.0
        if self._n_extra:
            n_poly = (3 if self.degree == 1 else 6)
            n_lid = 1 if self.use_lid else 0
            diag[n_poly + n_lid:] = lam * self.pose_ridge_factor
        return np.diag(diag)

    def _solve(self, raw: np.ndarray, targets: np.ndarray, lam: float, pose_ref: np.ndarray | None = None) -> None:
        self._mu = raw.mean(axis=0)
        self._sigma = raw.std(axis=0)
        self._sigma[self._sigma < 1e-9] = 1.0
        if self.use_head_pose:
            pc = self._pose_col
            self._mu[pc:pc + 4] = np.median(raw[:, pc:pc + 4], axis=0) if pose_ref is None else pose_ref
            self._sigma[pc:pc + 4] = POSE_SCALES
        z = (raw - self._mu) / self._sigma
        self._zmin, self._zmax = z[:, :2].min(axis=0), z[:, :2].max(axis=0)

        # Desacoplamento biomecânico em 2 estágios:
        # Quando há variação de pose ativa, o modelo ocular puro é ajustado nas amostras de repouso,
        # eliminando colinearidade e sobre-ajuste cruzado entre cabeça e olhos.
        if self.use_head_pose:
            pc = self._pose_col
            pose_dev = np.hypot(raw[:, pc] - self._mu[pc], raw[:, pc + 1] - self._mu[pc + 1])
            at_rest = pose_dev < 0.035

            if np.sum(at_rest) >= 8 and np.sum(~at_rest) >= 2:
                h_r, v_r = z[at_rest, 0], z[at_rest, 1]
                cols_r = [np.ones(len(h_r)), h_r, v_r]
                if self.degree == 2:
                    if getattr(self, "basis", "symmetric") == "symmetric":
                        cols_r += [h_r * v_r, h_r * np.abs(h_r), v_r * np.abs(v_r)]
                    else:
                        cols_r += [h_r * v_r, h_r ** 2, v_r ** 2]
                if self.use_lid:
                    cols_r.append(z[at_rest, 2])
                X_eye_r = np.column_stack(cols_r)

                pen_eye = np.full(X_eye_r.shape[1], lam)
                pen_eye[0] = 0.0
                beta_eye = np.linalg.solve(X_eye_r.T @ X_eye_r + np.diag(pen_eye), X_eye_r.T @ targets[at_rest])

                # Estágio 2: Resíduo de predição do olho para todas as amostras
                h_all, v_all = z[:, 0], z[:, 1]
                cols_all = [np.ones(len(h_all)), h_all, v_all]
                if self.degree == 2:
                    if getattr(self, "basis", "symmetric") == "symmetric":
                        cols_all += [h_all * v_all, h_all * np.abs(h_all), v_all * np.abs(v_all)]
                    else:
                        cols_all += [h_all * v_all, h_all ** 2, v_all ** 2]
                if self.use_lid:
                    cols_all.append(z[:, 2])
                X_eye_all = np.column_stack(cols_all)

                res = targets - X_eye_all @ beta_eye

                # Projeção ortogonal física das variáveis de cabeça (zero cross-talk)
                zh = z[:, pc:pc + 4]
                X_hx = zh[:, [0, 2]]
                pen_hx = np.diag([lam * self.pose_ridge_factor, lam * self.pose_ridge_factor])
                b_hx = np.linalg.solve(X_hx.T @ X_hx + pen_hx, X_hx.T @ res[:, 0])

                X_hy = zh[:, [1, 3]]
                pen_hy = np.diag([lam * self.pose_ridge_factor, lam * self.pose_ridge_factor])
                b_hy = np.linalg.solve(X_hy.T @ X_hy + pen_hy, X_hy.T @ res[:, 1])

                beta_head = np.zeros((4, 2))
                beta_head[0, 0] = np.clip(b_hx[0], -self.head_limit_px, self.head_limit_px)
                beta_head[2, 0] = np.clip(b_hx[1], -self.head_limit_px, self.head_limit_px)
                beta_head[1, 1] = np.clip(b_hy[0], -self.head_limit_px, self.head_limit_px)
                beta_head[3, 1] = np.clip(b_hy[1], -self.head_limit_px, self.head_limit_px)

                self._beta = np.vstack([beta_eye, beta_head])
                return

        x = self._design(raw)
        beta = np.linalg.solve(x.T @ x + self._penalty(lam, x.shape[1]), x.T @ targets)
        if self.use_head_pose:
            # Limite físico: |Δpx| por unidade de z (= 0,1 rad / 0,1 adim.) ≤ head_limit_px.
            first = x.shape[1] - 4
            beta[first:] = np.clip(beta[first:], -self.head_limit_px, self.head_limit_px)
        self._beta = beta

    def _loo(self, raw: np.ndarray, targets: np.ndarray, idx: np.ndarray, points: np.ndarray, lam: float,
             pose_ref: np.ndarray | None) -> float:
        errs = []
        for p in points:
            train = idx != p
            self._solve(raw[train], targets[train], lam, pose_ref)
            pred = self._design(raw[~train], clip=True) @ self._beta
            errs.append(np.linalg.norm(pred.mean(axis=0) - targets[~train][0]))
        return float(np.mean(errs))

    def _candidates(self, n_points: int, pose_ok: bool, lid_ok: bool):
        max_deg = self.degree
        for degree in range(1, max_deg + 1):
            bases = ("symmetric", "quadratic") if degree == 2 else ("linear",)
            for basis in bases:
                for lid in ((False, True) if (self.use_lid and lid_ok) else (False,)):
                    for pose in ((False, True) if (self.use_head_pose and pose_ok) else (False,)):
                        terms = (3 if degree == 1 else 6) + (1 if lid else 0) + (4 if pose else 0)
                        if n_points >= terms + 3:
                            yield degree, lid, pose, basis

    def fit(self, samples: list[CalibrationSample], pose_ref: tuple[float, float, float, float] | None = None) -> CalibrationReport:
        a = self._arrays(samples)
        idx = a[:, 6].astype(int)
        targets = a[:, 4:6]
        points = np.unique(idx)
        ref = None if pose_ref is None else np.asarray(pose_ref, dtype=float)

        max_cfg = {"degree": self.degree, "lid": self.use_lid, "pose": self.use_head_pose}
        if self.use_head_pose:
            pose_ok = bool(np.any(a[:, [2, 3, 8, 9]].std(axis=0) >= MIN_POSE_STD))  # alguma dimensão de pose varia
        else:
            pose_ok = False
        lid_ok = bool(self.use_lid and a[:, 7].std() > 1e-6)

        loo: float | None = None
        if self.auto_select:
            best = None
            for degree, lid, pose, basis in self._candidates(len(points), pose_ok, lid_ok):
                self.degree, self.use_lid, self.use_head_pose, self.basis = degree, lid, pose, basis
                raw = self._raw(a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 7], a[:, 8], a[:, 9])
                for lam in LAMBDA_GRID:
                    err = self._loo(raw, targets, idx, points, lam, ref)
                    if best is None or err < best[0]:
                        best = (err, degree, lid, pose, basis, lam)
            if best is None:  # poucos pontos: modelo linear simples
                self.degree, self.use_lid, self.use_head_pose, self.basis = 1, False, False, "linear"
                if len(points) < self.num_terms:
                    raise ValueError(f"Pontos insuficientes ({len(points)}) para o modelo mínimo.")
                best = (None, 1, False, False, "linear", max(LAMBDA_GRID[1], self.ridge_lambda))
            loo, self.degree, self.use_lid, self.use_head_pose, self.basis, self.ridge_lambda = best
        else:
            min_required = self.num_terms - (4 if self.use_head_pose else 0) - (1 if self.use_lid else 0)
            if len(points) < min_required:
                raise ValueError(f"Pontos insuficientes ({len(points)}) para grau {self.degree}.")
            if (self.use_head_pose and len(points) < self.num_terms) or not pose_ok:
                self.use_head_pose = False
            if self.use_lid and len(points) < self.num_terms:
                self.use_lid = False
            raw = self._raw(a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 7], a[:, 8], a[:, 9])
            if len(points) > self.num_terms:
                loo = self._loo(raw, targets, idx, points, self.ridge_lambda, ref)

        raw = self._raw(a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 7], a[:, 8], a[:, 9])
        self._solve(raw, targets, self.ridge_lambda, ref)
        if self.use_head_pose:
            pc = self._pose_col
            self._pose_max = np.abs(raw[:, pc:pc + 2] - self._mu[pc:pc + 2]).max(axis=0)
        pred = self._design(raw, clip=True) @ self._beta
        per_point = {int(p): float(np.linalg.norm(pred[idx == p].mean(axis=0) - targets[idx == p][0])) for p in points}
        return CalibrationReport(
            mean_error_px=float(np.mean(list(per_point.values()))),
            loo_mean_error_px=loo,
            per_point_error_px=per_point,
            num_samples=len(samples),
            num_points=len(points),
            model={"degree": self.degree, "lid": self.use_lid, "head_pose": self.use_head_pose,
                   "basis": self.basis, "lambda": self.ridge_lambda, "requested": max_cfg},
        )

    # ----------------------------------------------------------------- predict
    def predict_unbiased(self, f: GazeFeatures) -> np.ndarray:
        return self.predict_raw(f.h, f.v, f.yaw, f.pitch, apply_offset=False, lid=f.lid, hx=f.head_x, hy=f.head_y)

    def predict(self, f: GazeFeatures) -> np.ndarray:
        return self.predict_raw(f.h, f.v, f.yaw, f.pitch, apply_offset=True, lid=f.lid, hx=f.head_x, hy=f.head_y)

    def predict_raw(self, h: float, v: float, yaw: float = 0.0, pitch: float = 0.0, apply_offset: bool = True,
                    lid: float = 0.0, hx: float = 0.0, hy: float = 0.0) -> np.ndarray:
        if self._beta is None:
            raise NotFittedError("GazeMapper não calibrado.")
        raw = self._raw(h, v, yaw, pitch, lid, hx, hy)
        p = (self._design(raw, clip=True) @ self._beta)[0]
        w, hgt = self.screen_size
        if apply_offset:
            center = np.array([w, hgt], dtype=float) / 2.0
            p = center + self.scale * (p - center) + self.offset
        return np.array([np.clip(p[0], 0, w - 1), np.clip(p[1], 0, hgt - 1)])

    def pose_shift(self, prev: GazeFeatures, cur: GazeFeatures) -> np.ndarray:
        """Deslocamento em pixels explicado só pela mudança de pose entre dois frames."""
        if not self.use_head_pose or self._beta is None:
            return np.zeros(2)
        a = self.predict_raw(cur.h, cur.v, cur.yaw, cur.pitch, False, cur.lid, cur.head_x, cur.head_y)
        b = self.predict_raw(cur.h, cur.v, prev.yaw, prev.pitch, False, cur.lid, prev.head_x, prev.head_y)
        return a - b

    def pose_out_of_range(self, f: GazeFeatures, factor: float = 1.5, floor_rad: float = 0.087) -> bool:
        """True se a cabeça está muito além da faixa de pose vista na calibração."""
        if not self.use_head_pose or self._mu is None:
            return False
        pc = self._pose_col
        limit = factor * np.maximum(self._pose_max, floor_rad)
        dev = np.abs(np.array([f.yaw, f.pitch]) - self._mu[pc:pc + 2])
        return bool(np.any(dev > limit))

    # ----------------------------------------------------------- serialization
    def to_dict(self) -> dict:
        if self._beta is None:
            raise NotFittedError("GazeMapper não calibrado.")
        return {
            "version": self.VERSION,
            "degree": self.degree,
            "basis": getattr(self, "basis", "symmetric"),
            "ridge_lambda": self.ridge_lambda,
            "use_head_pose": self.use_head_pose,
            "use_lid": self.use_lid,
            "pose_ridge_factor": self.pose_ridge_factor,
            "head_limit_px": self.head_limit_px,
            "screen_size": list(self.screen_size),
            "offset": self.offset.tolist(),
            "scale": self.scale.tolist(),
            "mu": np.asarray(self._mu).tolist(),
            "sigma": np.asarray(self._sigma).tolist(),
            "beta": np.asarray(self._beta).tolist(),
            "zmin": np.asarray(self._zmin).tolist(),
            "zmax": np.asarray(self._zmax).tolist(),
            "pose_max": np.asarray(self._pose_max).tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GazeMapper":
        if d.get("version") != cls.VERSION:
            raise ValueError("Versão de mapeador incompatível.")
        offset = tuple(d.get("offset", [0.0, 0.0]))
        basis = d.get("basis", "symmetric")
        m = cls(d["degree"], d["ridge_lambda"], d["use_head_pose"], tuple(d["screen_size"]), offset=offset,
                use_lid=d.get("use_lid", False), pose_ridge_factor=d.get("pose_ridge_factor", 10.0),
                head_limit_px=d.get("head_limit_px", 450.0), basis=basis)
        m.scale = np.array(d.get("scale", [1.0, 1.0]), dtype=float)
        m._mu = np.array(d["mu"], dtype=float)
        m._sigma = np.array(d["sigma"], dtype=float)
        m._beta = np.array(d["beta"], dtype=float)
        m._zmin = np.array(d["zmin"], dtype=float) if "zmin" in d else None
        m._zmax = np.array(d["zmax"], dtype=float) if "zmax" in d else None
        m._pose_max = np.array(d.get("pose_max", [0.0, 0.0]), dtype=float)
        if m._beta.shape != (m.num_terms, 2):
            raise ValueError("Coeficientes com formato inválido.")
        return m
