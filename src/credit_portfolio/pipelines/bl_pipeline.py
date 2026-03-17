"""V2 Pipeline: data -> HMM -> BL -> optimizer -> charts."""

import numpy as np
import pandas as pd

from credit_portfolio.config import load_config, resolve_csv_path, resolve_output_dir
from credit_portfolio.log import get_logger
from credit_portfolio.data.loader import load
from credit_portfolio.data.constants import (
    DURATIONS, OMEGA_SCALE, IG_ASSETS, IG_MARKET_WEIGHTS_ARRAY,
    ASSET_LABELS, PIPELINE_DELTA, PIPELINE_TAU, COV_WINDOW,
    DEFAULT_BOND_COUNT, JUNE_SHOCK, BL_RIDGE_PENALTY,
    OAS_PCT_TO_BP, MONTHS_PER_YEAR,
)
from credit_portfolio.models.hmm_regime import fit_hmm, get_current_regime, regime_summary
from credit_portfolio.models.black_litterman import BLResult
from credit_portfolio.charts.regime import chart_hmm_regimes
from credit_portfolio.charts.empirical import (
    chart_value_signal, chart_momentum_signal, chart_quality_sharpe,
)
from credit_portfolio.charts.bl_charts import chart_bl_posterior

logger = get_logger(__name__)

BANNER = "=" * 62


def _run_inline_bl(df: pd.DataFrame, hmm_result) -> dict:
    """Run BL using trailing 3m OAS trend as views (same logic as pipeline.py)."""
    n = len(IG_ASSETS)
    regime_info = get_current_regime(hmm_result)

    # Covariance from real monthly excess returns
    ret_data = {}
    for a in IG_ASSETS:
        if a in df.columns:
            dur = DURATIONS.get(a, 7.0)
            ret_data[a] = -dur * df[a].diff(1) + df[a] / MONTHS_PER_YEAR
    Sigma = pd.DataFrame(ret_data).dropna().tail(COV_WINDOW).cov().values

    Pi = PIPELINE_DELTA * Sigma @ IG_MARKET_WEIGHTS_ARRAY

    Q_vec = np.zeros(n)
    oas_chg_bp = {}
    for i, a in enumerate(IG_ASSETS):
        if a in df.columns:
            chg_3m = float(df[a].diff(3).iloc[-1])
            dur = DURATIONS.get(a, 7.0)
            Q_vec[i] = -dur * chg_3m / OAS_PCT_TO_BP
            oas_chg_bp[a] = round(chg_3m * OAS_PCT_TO_BP, 1)

    P_mat = np.eye(n)
    omega_scale = hmm_result.omega_scale
    tS = PIPELINE_TAU * Sigma
    Omega = np.diag([omega_scale * PIPELINE_TAU * float(P_mat[i] @ tS @ P_mat[i])
                     for i in range(n)]) + np.eye(n) * BL_RIDGE_PENALTY

    tS_inv = np.linalg.inv(tS + np.eye(n) * BL_RIDGE_PENALTY)
    Om_inv = np.linalg.inv(Omega)
    A = tS_inv + P_mat.T @ Om_inv @ P_mat
    b = tS_inv @ Pi + P_mat.T @ Om_inv @ Q_vec

    try:
        mu_BL = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        mu_BL = Pi.copy()

    return {
        "Pi": Pi, "mu_BL": mu_BL, "Q_vec": Q_vec, "Sigma": Sigma,
        "omega_scale": omega_scale, "oas_chg_bp": oas_chg_bp,
        "regime_info": regime_info,
    }


def run(output_dir: str | None = None):
    cfg = load_config()
    csv_path = resolve_csv_path(cfg)
    if output_dir is None:
        output_dir = str(resolve_output_dir(cfg))

    logger.info(BANNER)
    logger.info(" THREE-LAYER FACTOR CREDIT PIPELINE")
    logger.info(" HMM  |  Black-Litterman  |  CVXPY Optimizer")
    logger.info(BANNER)

    logger.info("Loading real FRED data...")
    df = load(str(csv_path))
    logger.info("%d monthly observations", len(df))

    logger.info("Fitting HMM (3 states, full covariance)...")
    hmm_result = fit_hmm(df)
    regime_info = get_current_regime(hmm_result)
    logger.info("Current regime: %s", hmm_result.current_label)

    logger.info("Regime statistics:")
    logger.info(regime_summary(hmm_result).to_string())

    logger.info("Generating HMM regime chart...")
    chart_hmm_regimes(df, hmm_result, output_dir)

    logger.info("Running Black-Litterman...")
    bl = _run_inline_bl(df, hmm_result)
    Pi, mu_BL, Q_vec = bl["Pi"], bl["mu_BL"], bl["Q_vec"]

    logger.info(" %6s  %10s  %10s  %10s", "Asset", "Prior Pi", "View Q", "Posterior")
    logger.info(" %s", "-" * 40)
    for i, a in enumerate(IG_ASSETS):
        lbl = ASSET_LABELS[a]
        logger.info(" %6s  %+9.3f%%  %+9.3f%%  %+9.3f%%",
                     lbl, Pi[i]*100, Q_vec[i]*100, mu_BL[i]*100)

    logger.info("Generating BL posterior chart...")
    chart_bl_posterior(
        IG_ASSETS, Pi, mu_BL, Q_vec,
        hmm_result.current_label, hmm_result.omega_scale, output_dir
    )

    # Run factor optimizer layer
    try:
        from credit_portfolio.data.universe import build_universe
        from credit_portfolio.optimizers.factor_tilt import optimise, OptConfig

        df_t0 = build_universe(n=DEFAULT_BOND_COUNT)
        df_t1 = build_universe(n=DEFAULT_BOND_COUNT, shock=JUNE_SHOCK)
        config = OptConfig()
        r0 = optimise(df_t0, config=config)
        r1 = optimise(df_t1, prior_w=r0.weights.values, config=config)
        logger.info("March optimisation: %s", r0.status)
        logger.info("June  optimisation: %s", r1.status)
    except Exception as e:
        logger.warning("Optimizer layer skipped: %s", e)

    logger.info("Generating empirical charts...")
    chart_value_signal(df, output_dir)
    chart_momentum_signal(df, output_dir)
    chart_quality_sharpe(df, output_dir)

    logger.info(BANNER)
    logger.info(" PIPELINE COMPLETE")
    logger.info(" Charts saved to %s/", output_dir)
    logger.info(BANNER)
