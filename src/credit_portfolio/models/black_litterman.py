"""
Black-Litterman model for credit portfolio construction.

Implements the BL Bayesian framework: equilibrium prior + views -> posterior.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from credit_portfolio.data.constants import (
    MARKET_WEIGHTS, MARKET_WEIGHTS_WITH_HY, LAMBDA_RISK_AVERSION, DURATIONS,
    WITHIN_BUCKET_CORR, MIN_PERIODS_COV,
    BL_RIDGE_PENALTY, MONTHS_PER_YEAR, QUARTERLY_TO_ANNUAL, OAS_PCT_TO_BP,
)


@dataclass
class BLResult:
    mu_prior   : np.ndarray
    mu_bl      : np.ndarray
    sigma_bl   : np.ndarray
    sigma_hist : np.ndarray
    assets     : list
    tau        : float
    regime     : str
    weights_mkt: np.ndarray


def _build_covariance(df: pd.DataFrame) -> np.ndarray:
    """
    Build a covariance matrix from bond-level spread_vol_1y data.

    Uses diagonal variances from spread volatility with a constant
    correlation overlay to produce a positive-definite matrix.

    Parameters
    ----------
    df : DataFrame with 'spread_vol_1y' column (annualised spread vol in bp)

    Returns
    -------
    Positive-definite covariance matrix (n x n)
    """
    vols = df["spread_vol_1y"].values.astype(float)
    # Convert bp vol to decimal (100bp = 1%)
    vols_dec = vols / 100.0
    n = len(vols_dec)

    rho = WITHIN_BUCKET_CORR
    cov = np.outer(vols_dec, vols_dec) * rho
    np.fill_diagonal(cov, vols_dec ** 2)

    # Ensure positive definite with small ridge
    cov += np.eye(n) * BL_RIDGE_PENALTY
    return cov


def compute_historical_returns(monthly: pd.DataFrame,
                               assets: list) -> pd.DataFrame:
    """
    Compute monthly excess return proxies for each rating bucket.

    r_i(t) = OAS_i(t-1)/12 - duration_i * delta_OAS_i(t)/100
    """
    oas_cols = {
        "AAA": "oas_aaa", "AA": "oas_aa",
        "A"  : "oas_a",   "BBB": "oas_bbb", "HY": "oas_hy"
    }

    returns = {}
    for asset in assets:
        col = oas_cols.get(asset)
        if col is None or col not in monthly.columns:
            continue
        dur = DURATIONS.get(asset, 6.5)
        oas = monthly[col]
        carry = oas.shift(1) / MONTHS_PER_YEAR / OAS_PCT_TO_BP
        spread_chg = oas.diff(1)
        price_ret = dur * (-spread_chg / OAS_PCT_TO_BP)
        ret = (carry + price_ret).dropna()
        returns[asset] = ret

    return pd.DataFrame(returns).dropna()


def compute_covariance(ret_df: pd.DataFrame, min_periods: int = MIN_PERIODS_COV) -> np.ndarray:
    """Compute annualised covariance matrix from monthly return proxies."""
    cov_monthly = ret_df.cov()
    return (cov_monthly * MONTHS_PER_YEAR).values


def compute_equilibrium_prior(sigma: np.ndarray, w_mkt: np.ndarray,
                              lam: float = LAMBDA_RISK_AVERSION) -> np.ndarray:
    """Reverse-optimisation: pi = lambda * Sigma * w_mkt."""
    return lam * sigma @ w_mkt


def build_views_matrices(views: dict, assets: list,
                         sigma: np.ndarray, tau: float) -> tuple:
    """Construct P, Q, Omega from Prophet views."""
    n = len(assets)
    asset_idx = {a: i for i, a in enumerate(assets)}

    view_assets = [a for a in assets if a in views]
    k = len(view_assets)

    if k == 0:
        return None, None, None

    P = np.zeros((k, n))
    Q = np.zeros(k)

    for i, asset in enumerate(view_assets):
        j = asset_idx[asset]
        P[i, j] = 1.0
        Q[i] = views[asset]["expected_return"] * QUARTERLY_TO_ANNUAL  # annualise from 3m

    Omega = tau * (P @ sigma @ P.T)
    Omega += np.eye(k) * BL_RIDGE_PENALTY

    return P, Q, Omega


def black_litterman_posterior(sigma: np.ndarray, pi: np.ndarray,
                              P: np.ndarray, Q: np.ndarray,
                              Omega: np.ndarray, tau: float) -> tuple:
    """
    BL Master Formula:
        mu_BL = [(tau*Sigma)^-1 + P'*Omega^-1*P]^-1 * [(tau*Sigma)^-1*pi + P'*Omega^-1*Q]
        Sigma_BL = Sigma + [(tau*Sigma)^-1 + P'*Omega^-1*P]^-1
    """
    tau_sigma = tau * sigma
    tau_sigma_inv = np.linalg.inv(tau_sigma)
    Omega_inv = np.linalg.inv(Omega)

    M_inv = tau_sigma_inv + P.T @ Omega_inv @ P
    M = np.linalg.inv(M_inv)

    mu_bl = M @ (tau_sigma_inv @ pi + P.T @ Omega_inv @ Q)
    sigma_bl = sigma + M

    return mu_bl, sigma_bl


def run_black_litterman(monthly: pd.DataFrame, views: dict,
                        regime_info: dict,
                        include_hy: bool = False) -> BLResult:
    """Full Black-Litterman pipeline."""
    assets = list(MARKET_WEIGHTS_WITH_HY.keys()) if include_hy \
             else list(MARKET_WEIGHTS.keys())
    w_mkt = np.array([
        (MARKET_WEIGHTS_WITH_HY if include_hy else MARKET_WEIGHTS)[a]
        for a in assets
    ])
    w_mkt /= w_mkt.sum()

    tau = regime_info["bl_tau"]
    regime = regime_info["regime"]

    ret_df = compute_historical_returns(monthly, assets)
    sigma = compute_covariance(ret_df)

    pi = compute_equilibrium_prior(sigma, w_mkt)

    P, Q, Omega = build_views_matrices(views, assets, sigma, tau)

    if P is None:
        return BLResult(
            mu_prior=pi, mu_bl=pi, sigma_bl=sigma, sigma_hist=sigma,
            assets=assets, tau=tau, regime=regime, weights_mkt=w_mkt
        )

    mu_bl, sigma_bl = black_litterman_posterior(sigma, pi, P, Q, Omega, tau)

    return BLResult(
        mu_prior=pi, mu_bl=mu_bl, sigma_bl=sigma_bl, sigma_hist=sigma,
        assets=assets, tau=tau, regime=regime, weights_mkt=w_mkt
    )
