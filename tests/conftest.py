"""Shared fixtures for all tests."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path


@pytest.fixture
def sample_universe():
    """60-bond DataFrame from build_universe."""
    from credit_portfolio.data.universe import build_universe
    return build_universe(n=60)


@pytest.fixture
def small_universe():
    """10-bond universe for fast optimizer tests."""
    from credit_portfolio.data.universe import build_universe
    return build_universe(n=10)


@pytest.fixture
def sample_universe_pair():
    """(df_t0, df_t1) with JUNE_SHOCK applied."""
    from credit_portfolio.data.universe import build_universe, JUNE_SHOCK
    df_t0 = build_universe(n=60)
    df_t1 = build_universe(n=60, shock=JUNE_SHOCK)
    return df_t0, df_t1


@pytest.fixture
def sample_monthly_data():
    """3 years of simulated monthly OAS (no file I/O)."""
    rng = np.random.default_rng(123)
    dates = pd.date_range("2020-01-31", periods=36, freq="ME")
    n = len(dates)
    oas_ig = 1.0 + np.cumsum(rng.normal(0, 0.05, n))
    oas_ig = np.clip(oas_ig, 0.3, 6.0)
    df = pd.DataFrame({
        "oas_ig" : oas_ig,
        "oas_aaa": oas_ig * 0.3,
        "oas_aa" : oas_ig * 0.6,
        "oas_a"  : oas_ig * 0.85,
        "oas_bbb": oas_ig * 1.4,
        "oas_hy" : oas_ig * 3.5,
        "tr_ig"  : 100 * np.cumprod(1 + rng.normal(0.003, 0.01, n)),
        "tr_bbb" : 100 * np.cumprod(1 + rng.normal(0.004, 0.012, n)),
        "tr_hy"  : 100 * np.cumprod(1 + rng.normal(0.005, 0.015, n)),
    }, index=dates)
    return df


@pytest.fixture
def sample_covariance_4x4():
    """Known PD 4x4 covariance matrix."""
    rng = np.random.default_rng(42)
    A = rng.normal(size=(4, 4))
    cov = A @ A.T / 16 + np.eye(4) * 0.01
    return cov


@pytest.fixture
def sample_bl_result(sample_covariance_4x4):
    """Pre-computed BLResult for optimizer tests."""
    from credit_portfolio.models.black_litterman import BLResult
    n = 4
    assets = ["AAA", "AA", "A", "BBB"]
    w_mkt = np.array([0.04, 0.12, 0.34, 0.50])
    sigma = sample_covariance_4x4
    mu_prior = 2.5 * sigma @ w_mkt
    mu_bl = mu_prior * 1.1  # slightly tilted
    sigma_bl = sigma + np.eye(n) * 0.001
    return BLResult(
        mu_prior=mu_prior, mu_bl=mu_bl, sigma_bl=sigma_bl,
        sigma_hist=sigma, assets=assets, tau=0.025,
        regime="NORMAL", weights_mkt=w_mkt,
    )


@pytest.fixture
def sample_opt_result_pair(sample_universe_pair):
    """(result_t0, result_t1) for attribution tests."""
    from credit_portfolio.optimizers.factor_tilt import optimise
    df_t0, df_t1 = sample_universe_pair
    r0 = optimise(df_t0)
    r1 = optimise(df_t1, prior_w=r0.weights.values)
    return r0, r1


@pytest.fixture
def fred_csv_path():
    """Real CSV path or pytest.skip."""
    p = Path(__file__).resolve().parents[1] / "data" / "fred_credit_spreads.csv"
    if not p.exists():
        pytest.skip("fred_credit_spreads.csv not found")
    return str(p)


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Temp dir for chart files."""
    return str(tmp_path)
