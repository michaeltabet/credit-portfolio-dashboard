"""
V3 Pipeline: data -> HMM -> BL -> ML Factor Model -> Optimizer -> Charts.

Extends the BL pipeline by adding an ML layer that trains tree-based models
on historical factor Z-scores, uses SHAP to derive time-varying factor weights,
and feeds these into the CVXPY optimizer.
"""

import numpy as np
import pandas as pd
import os

from credit_portfolio.config import load_config, resolve_csv_path, resolve_output_dir
from credit_portfolio.log import get_logger
from credit_portfolio.data.loader import load
from credit_portfolio.data.universe import build_universe, compute_forward_returns
from credit_portfolio.data.constants import (
    IG_ASSETS, IG_MARKET_WEIGHTS_ARRAY, ASSET_LABELS,
    PIPELINE_DELTA, PIPELINE_TAU, COV_WINDOW, DURATIONS,
    DEFAULT_BOND_COUNT, JUNE_SHOCK,
    ML_FEATURES, ML_MODEL_TYPE, ML_BL_BLEND_WEIGHT, ML_USE_SHAP_WEIGHTS,
    OPT_FACTOR_WEIGHTS, OMEGA_SCALE,
    BL_RIDGE_PENALTY, OAS_PCT_TO_BP, MONTHS_PER_YEAR, ML_NUMERIC_TOL,
    ML_HISTORICAL_PERIODS, ML_RANDOM_SEED, ML_SHOCK_START_PERIOD,
    ML_SHOCK_SEED_OFFSET, ML_SHOCK_BOND_STEP, ML_SHOCK_STD_DEV,
    ML_TARGET_COL,
)
from credit_portfolio.models.hmm_regime import fit_hmm, get_current_regime, regime_summary
from credit_portfolio.models.ml_factor_model import (
    train_and_predict, blend_with_bl, MLFactorResult,
)
from credit_portfolio.optimizers.factor_tilt import optimise, OptConfig
from credit_portfolio.charts.regime import chart_hmm_regimes
from credit_portfolio.charts.bl_charts import chart_bl_posterior
from credit_portfolio.charts.ml_charts import (
    chart_shap_summary, chart_shap_weights_over_time,
    chart_walk_forward_performance, chart_factor_weights_comparison,
)

logger = get_logger(__name__)

BANNER = "=" * 62


def _run_inline_bl(df: pd.DataFrame, hmm_result) -> dict:
    """Run BL using trailing 3m OAS trend as views (reused from bl_pipeline)."""
    n = len(IG_ASSETS)
    regime_info = get_current_regime(hmm_result)

    ret_data = {}
    for a in IG_ASSETS:
        if a in df.columns:
            dur = DURATIONS.get(a, 7.0)
            ret_data[a] = -dur * df[a].diff(1) + df[a] / MONTHS_PER_YEAR
    Sigma = pd.DataFrame(ret_data).dropna().tail(COV_WINDOW).cov().values

    Pi = PIPELINE_DELTA * Sigma @ IG_MARKET_WEIGHTS_ARRAY

    Q_vec = np.zeros(n)
    for i, a in enumerate(IG_ASSETS):
        if a in df.columns:
            chg_3m = float(df[a].diff(3).iloc[-1])
            dur = DURATIONS.get(a, 7.0)
            Q_vec[i] = -dur * chg_3m / OAS_PCT_TO_BP

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
        "omega_scale": omega_scale, "regime_info": regime_info,
    }


def _build_historical_panel(
    n_periods: int = ML_HISTORICAL_PERIODS,
    n_bonds: int = DEFAULT_BOND_COUNT,
    base_seed: int = ML_RANDOM_SEED,
) -> pd.DataFrame:
    """
    Build historical panel of bond-month observations for ML training.

    Each period gets a slightly different universe (varying seed) to simulate
    temporal variation. For production, replace with actual historical bond data.
    """
    panels = []
    for t in range(n_periods):
        seed = base_seed + t * 7
        shock = None
        # Apply stress-like shocks in later periods
        if t >= ML_SHOCK_START_PERIOD:
            rng = np.random.default_rng(seed + ML_SHOCK_SEED_OFFSET)
            shock = {
                f"BOND{i:03d}": {"spread_6m_chg": float(rng.normal(0, ML_SHOCK_STD_DEV))}
                for i in range(0, n_bonds, ML_SHOCK_BOND_STEP)
            }

        df = build_universe(n=n_bonds, shock=shock, seed=seed)
        df[ML_TARGET_COL] = compute_forward_returns(df)
        df["date"] = pd.Timestamp("2023-01-01") + pd.DateOffset(months=t)
        panels.append(df)

    return pd.concat(panels, ignore_index=True)


def run(output_dir: str | None = None, model_type: str | None = None):
    cfg = load_config()
    csv_path = resolve_csv_path(cfg)
    if output_dir is None:
        output_dir = str(resolve_output_dir(cfg))
    if model_type is None:
        model_type = ML_MODEL_TYPE

    os.makedirs(output_dir, exist_ok=True)

    logger.info(BANNER)
    logger.info(" ML-ENHANCED CREDIT FACTOR PIPELINE (V3)")
    logger.info(" HMM | Black-Litterman | ML Factor Model | CVXPY")
    logger.info(BANNER)

    # ── Step 1: Load data + HMM ─────────────────────────────────
    logger.info("[1/7] Loading real FRED data...")
    df = load(str(csv_path))
    logger.info("      %d monthly observations", len(df))

    logger.info("[2/7] Fitting HMM regime model...")
    hmm_result = fit_hmm(df)
    regime = hmm_result.current_label
    logger.info("      Current regime: %s", regime)
    logger.info("Regime statistics:")
    logger.info(regime_summary(hmm_result).to_string())
    chart_hmm_regimes(df, hmm_result, output_dir)

    # ── Step 2: Black-Litterman ──────────────────────────────────
    logger.info("[3/7] Running Black-Litterman...")
    bl = _run_inline_bl(df, hmm_result)
    Pi, mu_BL, Q_vec = bl["Pi"], bl["mu_BL"], bl["Q_vec"]

    logger.info("      %6s  %10s  %10s  %10s", "Asset", "Prior", "View", "Posterior")
    logger.info("      %s", "-" * 40)
    for i, a in enumerate(IG_ASSETS):
        lbl = ASSET_LABELS[a]
        logger.info("      %6s  %+9.3f%%  %+9.3f%%  %+9.3f%%",
                     lbl, Pi[i]*100, Q_vec[i]*100, mu_BL[i]*100)

    chart_bl_posterior(
        IG_ASSETS, Pi, mu_BL, Q_vec,
        hmm_result.current_label, hmm_result.omega_scale, output_dir,
    )

    # ── Step 3: Build universe ──────────────────────────────────
    logger.info("[4/7] Building bond universe...")
    df_current = build_universe(n=DEFAULT_BOND_COUNT)
    df_current[ML_TARGET_COL] = compute_forward_returns(df_current)

    avail_features = [f for f in ML_FEATURES if f in df_current.columns]
    logger.info("      %d bonds, %d ML features: %s", len(df_current), len(avail_features), avail_features)

    # ── Step 4: ML Factor Model ─────────────────────────────────
    logger.info("[5/7] Training ML factor model (%s)...", model_type)
    panel = _build_historical_panel()

    ml_result = train_and_predict(
        panel=panel,
        current_universe=df_current,
        features=avail_features,
        model_type=model_type,
        regime=regime,
    )

    logger.info("      OOS R-squared (mean): %.4f", ml_result.oos_r2_mean)
    logger.info("      OOS Rank IC  (mean): %.4f", ml_result.oos_ic_mean)
    logger.info("      SHAP factor weights:")
    for feat, weight in sorted(ml_result.shap_factor_weights.items(),
                                key=lambda x: -x[1]):
        logger.info("        %15s: %.3f", feat, weight)

    # ── Step 5: Optimize with SHAP weights ──────────────────────
    logger.info("[6/7] Running optimizers (fixed vs ML-weighted)...")

    if ML_USE_SHAP_WEIGHTS:
        # Map SHAP weights to optimizer factors (exclude z_carry, redistribute)
        shap_w = ml_result.shap_factor_weights
        opt_factors = list(OPT_FACTOR_WEIGHTS.keys())
        factor_weight_map = {f: shap_w.get(f, 0.0) for f in opt_factors}
        total = sum(factor_weight_map.values())
        if total > ML_NUMERIC_TOL:
            ml_factor_weights = {f: v / total for f, v in factor_weight_map.items()}
        else:
            ml_factor_weights = dict(OPT_FACTOR_WEIGHTS)

        config_ml = OptConfig(factor_weights=ml_factor_weights)
    else:
        config_ml = OptConfig()

    result_ml = optimise(df_current, config=config_ml)
    result_eq = optimise(df_current, config=OptConfig())

    logger.info("      ML-weighted factor tilt: %s (obj=%.4f)", result_ml.status, result_ml.objective_value)
    logger.info("      Fixed-weighted baseline: %s (obj=%.4f)", result_eq.status, result_eq.objective_value)

    if ML_USE_SHAP_WEIGHTS:
        logger.info("      ML factor weights used:")
        for f, w in sorted(ml_factor_weights.items(), key=lambda x: -x[1]):
            fixed_w = OPT_FACTOR_WEIGHTS.get(f, 0)
            logger.info("        %15s: %.3f (was %.3f)", f, w, fixed_w)

    # ── Step 6: Charts ──────────────────────────────────────────
    logger.info("[7/7] Generating ML charts...")

    chart_shap_summary(
        ml_result.shap_values_current,
        ml_result.feature_names,
        df_current,
        output_dir,
    )
    chart_shap_weights_over_time(ml_result.shap_weights_history, output_dir)
    chart_walk_forward_performance(ml_result.walk_forward_folds, output_dir)
    chart_factor_weights_comparison(
        OPT_FACTOR_WEIGHTS,
        ml_result.shap_factor_weights,
        output_dir,
    )

    logger.info(BANNER)
    logger.info(" ML PIPELINE COMPLETE")
    logger.info(" Charts saved to %s/", output_dir)
    logger.info(BANNER)
