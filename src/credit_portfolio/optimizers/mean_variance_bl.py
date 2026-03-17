"""
BL-enhanced mean-variance portfolio optimizer.

Objective: max w'mu_BL - (lambda/2) * w'Sigma_BL * w
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from dataclasses import dataclass
from typing import Optional

from credit_portfolio.data.constants import (
    GAMMA_FACTOR, WITHIN_BUCKET_CORR,
    OPT_MAX_SECTOR_DEV, OPT_MAX_DUR_DEV, OPT_MAX_TURNOVER,
    OPT_MAX_SINGLE_NAME, OPT_QUALITY_FLOOR, LAMBDA_RISK_AVERSION,
    OPT_CONSTRAINT_TOL,
)


@dataclass
class BLOptResult:
    weights            : pd.Series
    active_weights     : pd.Series
    expected_returns   : pd.Series
    factor_exposures   : dict
    binding_constraints: list
    objective_value    : float
    status             : str
    regime             : str
    tau                : float
    n_excluded         : int


def map_bl_returns_to_bonds(
    df: pd.DataFrame,
    bl_result,
    ml_alpha: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Map BL bucket-level expected returns to individual bonds.

    If ml_alpha is provided, it replaces the linear GAMMA_FACTOR * z_composite
    with ML-derived bond-level alpha (from SHAP-based predictions).
    """
    bucket_map = dict(zip(bl_result.assets, bl_result.mu_bl))
    n = len(df)
    expected_returns = np.zeros(n)

    for i, row in df.iterrows():
        bucket_ret = bucket_map.get(row["rating"], np.mean(bl_result.mu_bl))
        if ml_alpha is not None:
            factor_alpha = ml_alpha[i]
        else:
            factor_alpha = GAMMA_FACTOR * row.get("z_composite", 0.0)
        expected_returns[i] = bucket_ret + factor_alpha

    return expected_returns


def optimise_bl(df: pd.DataFrame, bl_result, prior_w=None,
                max_sector_dev=OPT_MAX_SECTOR_DEV, max_dur_dev=OPT_MAX_DUR_DEV,
                max_turnover=OPT_MAX_TURNOVER, max_single_name=OPT_MAX_SINGLE_NAME,
                quality_floor=OPT_QUALITY_FLOOR, lambda_risk=LAMBDA_RISK_AVERSION) -> BLOptResult:
    """Mean-variance optimizer with BL expected returns."""
    eligible = df["quality_score"] >= quality_floor
    n_excl = (~eligible).sum()
    df_e = df[eligible].copy().reset_index(drop=True)
    n = len(df_e)

    bmark = np.ones(n) / n

    mu_i = map_bl_returns_to_bonds(df_e, bl_result)

    assets_list = bl_result.assets
    sigma_bl = bl_result.sigma_bl
    bucket_var = {
        a: sigma_bl[i, i] for i, a in enumerate(assets_list)
    }
    bond_var = np.array([
        bucket_var.get(row["rating"], np.mean(list(bucket_var.values())))
        for _, row in df_e.iterrows()
    ])
    rho = WITHIN_BUCKET_CORR
    bond_std = np.sqrt(bond_var)
    Sigma_bond = np.outer(bond_std, bond_std) * rho
    np.fill_diagonal(Sigma_bond, bond_var)

    w = cp.Variable(n, nonneg=True)
    portfolio_return = mu_i @ w
    portfolio_var = cp.quad_form(w, Sigma_bond)
    objective = cp.Maximize(portfolio_return - (lambda_risk / 2) * portfolio_var)

    constraints = [cp.sum(w) == 1, w <= max_single_name]

    for sector in df_e["sector"].unique():
        mask = (df_e["sector"] == sector).values
        bmark_s = bmark[mask].sum()
        constraints += [
            cp.sum(w[mask]) <= bmark_s + max_sector_dev,
            cp.sum(w[mask]) >= bmark_s - max_sector_dev,
        ]

    for db in df_e["duration_bucket"].unique():
        mask = (df_e["duration_bucket"] == db).values
        bmark_d = bmark[mask].sum()
        constraints += [
            cp.sum(w[mask]) <= bmark_d + max_dur_dev,
            cp.sum(w[mask]) >= bmark_d - max_dur_dev,
        ]

    if prior_w is not None:
        pw = np.zeros(n)
        for i, bid in enumerate(df_e["bond_id"].values):
            orig = df[df["bond_id"] == bid].index
            if len(orig) > 0 and orig[0] < len(prior_w):
                pw[i] = prior_w[orig[0]]
        if pw.sum() > 0:
            pw /= pw.sum()
        constraints += [cp.norm1(w - pw) / 2 <= max_turnover]

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
    exp_rets = pd.Series(mu_i, index=df_e["bond_id"].values)

    factor_exp = {
        c: float(np.dot(w_vals, df_e[c].values))
        for c in ["z_dts", "z_value", "z_momentum"]
        if c in df_e.columns
    }

    binding = []
    tol = OPT_CONSTRAINT_TOL
    if prior_w is not None:
        to = float(np.sum(np.abs(w_vals - pw)) / 2)
        if abs(to - max_turnover) < tol:
            binding.append(f"Turnover cap binding ({to:.1%})")
    if (w_vals > max_single_name - tol).any():
        hits = df_e["bond_id"].values[w_vals > max_single_name - tol]
        binding.append(f"Single-name cap: {', '.join(hits[:2])}")

    return BLOptResult(
        weights=w_series, active_weights=active,
        expected_returns=exp_rets, factor_exposures=factor_exp,
        binding_constraints=binding,
        objective_value=float(prob.value) if prob.value else 0.0,
        status=status,
        regime=bl_result.regime, tau=bl_result.tau,
        n_excluded=n_excl,
    )
