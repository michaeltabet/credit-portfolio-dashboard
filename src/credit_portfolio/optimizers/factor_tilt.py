"""
Factor-tilted credit portfolio optimizer using CVXPY.

Objective: maximise composite factor Z-score tilt.
Constraints: sector/duration neutrality, turnover cap, single-name cap, quality floor.
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from dataclasses import dataclass, field
from typing import Optional

from credit_portfolio.data.constants import (
    OPT_MAX_SECTOR_DEV, OPT_MAX_DUR_DEV, OPT_MAX_TURNOVER,
    OPT_MAX_SINGLE_NAME, OPT_QUALITY_FLOOR, OPT_FACTOR_WEIGHTS,
    OPT_CONSTRAINT_TOL,
)


@dataclass
class OptConfig:
    max_sector_dev : float = OPT_MAX_SECTOR_DEV
    max_dur_dev    : float = OPT_MAX_DUR_DEV
    max_turnover   : float = OPT_MAX_TURNOVER
    max_single_name: float = OPT_MAX_SINGLE_NAME
    quality_floor  : float = OPT_QUALITY_FLOOR
    factor_weights : dict = field(default_factory=lambda: dict(OPT_FACTOR_WEIGHTS))


@dataclass
class OptResult:
    weights            : pd.Series
    active_weights     : pd.Series
    factor_exposures   : dict
    binding_constraints: list
    objective_value    : float
    status             : str
    n_excluded         : int


def _benchmark_weights(df: pd.DataFrame) -> np.ndarray:
    return np.ones(len(df)) / len(df)


def optimise(
    df      : pd.DataFrame,
    prior_w : Optional[np.ndarray] = None,
    config  : OptConfig | None = None,
) -> OptResult:
    """Run the factor-tilted credit optimisation."""
    if config is None:
        config = OptConfig()

    eligible = df["quality_score"] >= config.quality_floor
    n_excluded = (~eligible).sum()
    df_e = df[eligible].copy().reset_index(drop=True)
    n = len(df_e)

    bmark = _benchmark_weights(df_e)

    score_cols = list(config.factor_weights.keys())
    fw = np.array([config.factor_weights[c] for c in score_cols])
    F = df_e[score_cols].values
    composite = F @ fw

    w = cp.Variable(n, nonneg=True)
    objective = cp.Maximize(composite @ w)
    constraints = [cp.sum(w) == 1]
    constraints += [w <= config.max_single_name]

    for sector in df_e["sector"].unique():
        mask = (df_e["sector"] == sector).values
        bmark_sec = bmark[mask].sum()
        constraints += [
            cp.sum(w[mask]) <= bmark_sec + config.max_sector_dev,
            cp.sum(w[mask]) >= bmark_sec - config.max_sector_dev,
        ]

    for db in df_e["duration_bucket"].unique():
        mask = (df_e["duration_bucket"] == db).values
        bmark_db = bmark[mask].sum()
        constraints += [
            cp.sum(w[mask]) <= bmark_db + config.max_dur_dev,
            cp.sum(w[mask]) >= bmark_db - config.max_dur_dev,
        ]

    if prior_w is not None:
        prior_aligned = np.zeros(n)
        prior_ids = df[eligible]["bond_id"].values
        for i, bid in enumerate(prior_ids):
            orig_idx = df[df["bond_id"] == bid].index
            if len(orig_idx) > 0 and orig_idx[0] < len(prior_w):
                prior_aligned[i] = prior_w[orig_idx[0]]
        if prior_aligned.sum() > 0:
            prior_aligned /= prior_aligned.sum()
        turnover = cp.norm1(w - prior_aligned) / 2
        constraints += [turnover <= config.max_turnover]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        w_vals = bmark
        status = prob.status or "infeasible"
    else:
        w_vals = np.array(w.value).clip(0)
        w_vals /= w_vals.sum()
        status = "optimal"

    w_series = pd.Series(w_vals, index=df_e["bond_id"].values)
    active = pd.Series(w_vals - bmark, index=df_e["bond_id"].values)

    factor_exp = {
        col: float(np.dot(w_vals, df_e[col].values))
        for col in score_cols
    }

    binding = []
    tol = OPT_CONSTRAINT_TOL

    for sector in df_e["sector"].unique():
        mask = (df_e["sector"] == sector).values
        port_sec = w_vals[mask].sum()
        bmark_sec = bmark[mask].sum()
        if abs(port_sec - bmark_sec - config.max_sector_dev) < tol:
            binding.append(f"Sector cap binding: {sector} (upper)")
        if abs(bmark_sec - port_sec - config.max_sector_dev) < tol:
            binding.append(f"Sector cap binding: {sector} (lower)")

    if prior_w is not None:
        to = float(np.sum(np.abs(w_vals - prior_aligned)) / 2)
        if abs(to - config.max_turnover) < tol:
            binding.append(f"Turnover cap binding ({to:.1%})")

    if (w_vals > config.max_single_name - tol).any():
        hits = df_e["bond_id"].values[w_vals > config.max_single_name - tol]
        binding.append(f"Single-name cap binding: {', '.join(hits[:3])}")

    return OptResult(
        weights=w_series,
        active_weights=active,
        factor_exposures=factor_exp,
        binding_constraints=binding,
        objective_value=float(prob.value) if prob.value else 0.0,
        status=status,
        n_excluded=n_excluded,
    )
