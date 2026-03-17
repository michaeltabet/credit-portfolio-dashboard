"""
Risk parity (Equal Risk Contribution) portfolio construction.

Allocates such that every bond contributes equally to total portfolio spread risk.
Uses _build_covariance from black_litterman.py (the previously broken import is now fixed).
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from dataclasses import dataclass
from typing import Optional

from credit_portfolio.models.black_litterman import _build_covariance
from credit_portfolio.data.constants import (
    RP_MAX_SINGLE_NAME, RP_MAX_SECTOR_DEV, OPT_QUALITY_FLOOR, RP_FACTOR_BLEND,
    RP_VOL_FLOOR, BL_RIDGE_PENALTY,
)


@dataclass
class RiskParityConfig:
    max_single_name: float = RP_MAX_SINGLE_NAME
    max_sector_dev : float = RP_MAX_SECTOR_DEV
    quality_floor  : float = OPT_QUALITY_FLOOR
    factor_blend   : float = RP_FACTOR_BLEND


@dataclass
class RiskParityResult:
    weights            : pd.Series
    risk_contributions : pd.Series
    total_portfolio_vol: float
    concentration_ratio: float
    status             : str
    n_excluded         : int


def _risk_contribution(w: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Compute marginal risk contributions."""
    port_vol = np.sqrt(w @ sigma @ w)
    if port_vol < RP_VOL_FLOOR:
        return np.ones(len(w)) / len(w)
    marginal = sigma @ w
    return w * marginal / port_vol


def optimise_risk_parity(
    df     : pd.DataFrame,
    config : Optional[RiskParityConfig] = None,
    sigma  : Optional[np.ndarray] = None,
) -> RiskParityResult:
    """
    Solve the ERC problem via log-barrier reformulation.

    min  w' * Sigma * w - (1/n) * sum_i log(w_i)
    s.t. w >= 0, sum(w) = 1
    """
    if config is None:
        config = RiskParityConfig()

    eligible = df["quality_score"] >= config.quality_floor
    n_excluded = (~eligible).sum()
    df_e = df[eligible].copy().reset_index(drop=True)
    n = len(df_e)

    if sigma is None:
        sigma = _build_covariance(df_e)
    else:
        sigma = sigma[np.ix_(eligible.values, eligible.values)]

    w = cp.Variable(n, pos=True)
    portfolio_var = cp.quad_form(w, sigma)
    log_barrier = (1.0 / n) * cp.sum(cp.log(w))
    objective = cp.Minimize(portfolio_var - log_barrier)

    constraints = [cp.sum(w) == 1]
    constraints += [w <= config.max_single_name]

    bmark = np.ones(n) / n
    for sector in df_e["sector"].unique():
        mask = (df_e["sector"] == sector).values
        bmark_sec = bmark[mask].sum()
        constraints += [
            cp.sum(w[mask]) <= bmark_sec + config.max_sector_dev,
            cp.sum(w[mask]) >= max(0.0, bmark_sec - config.max_sector_dev),
        ]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    if prob.status in ["optimal", "optimal_inaccurate"] and w.value is not None:
        w_vals = np.array(w.value).clip(BL_RIDGE_PENALTY)
        w_vals /= w_vals.sum()
        status = "optimal"
    else:
        vols = np.sqrt(np.diag(sigma))
        w_vals = (1.0 / vols)
        w_vals /= w_vals.sum()
        status = "fallback_inv_vol"

    w_series = pd.Series(w_vals, index=df_e["bond_id"].values)
    rc = _risk_contribution(w_vals, sigma)
    rc_series = pd.Series(rc / rc.sum(), index=df_e["bond_id"].values)

    port_vol = float(np.sqrt(w_vals @ sigma @ w_vals))
    herf = float(np.sum((rc / rc.sum()) ** 2))

    return RiskParityResult(
        weights=w_series,
        risk_contributions=rc_series,
        total_portfolio_vol=port_vol,
        concentration_ratio=herf,
        status=status,
        n_excluded=n_excluded,
    )


def compare_allocations(
    df       : pd.DataFrame,
    w_factor : pd.Series,
    w_rp     : pd.Series,
    w_bmark  : Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Compare factor tilt, risk parity, and equal weight allocations."""
    if w_bmark is None:
        n = len(df)
        w_bmark = pd.Series(np.ones(n) / n, index=df["bond_id"].values)

    rows = []
    for sector in sorted(df["sector"].unique()):
        bonds = df[df["sector"] == sector]["bond_id"].values
        rows.append({
            "sector"      : sector,
            "factor_tilt" : round(w_factor.reindex(bonds).fillna(0).sum() * 100, 1),
            "risk_parity" : round(w_rp.reindex(bonds).fillna(0).sum() * 100, 1),
            "equal_weight": round(w_bmark.reindex(bonds).fillna(0).sum() * 100, 1),
        })

    return pd.DataFrame(rows).sort_values("sector")
