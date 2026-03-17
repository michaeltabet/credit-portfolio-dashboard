"""
Rolling walk-forward ML backtest for bond-level factor portfolios.

Compares three strategies at each monthly rebalance:
  1. ML-weighted: factor weights from SHAP (retrained periodically)
  2. Fixed-weighted: static factor weights (DTS 50%, Value 25%, Momentum 25%)
  3. Benchmark: equal weight (1/n)

No look-ahead: at period t, ML trains on periods 0..t-1 only.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from credit_portfolio.data.universe import build_universe, compute_forward_returns
from credit_portfolio.data.constants import (
    DEFAULT_BOND_COUNT, OPT_FACTOR_WEIGHTS, ML_FEATURES,
    ML_MODEL_TYPE, ML_MIN_TRAIN_MONTHS,
    BT_N_PERIODS, BT_TC_BPS, BT_ML_RETRAIN_EVERY, BT_BASE_SEED,
    BT_MIN_ML_HISTORY,
)
from credit_portfolio.models.ml_factor_model import (
    build_model, compute_shap_factor_weights, _oversample_extremes,
)
from credit_portfolio.optimizers.factor_tilt import optimise, OptConfig
from credit_portfolio.log import get_logger

import shap

logger = get_logger(__name__)


@dataclass
class MLBacktestConfig:
    n_periods: int = BT_N_PERIODS
    n_bonds: int = DEFAULT_BOND_COUNT
    model_type: str = ML_MODEL_TYPE
    tc_bps: float = BT_TC_BPS
    ml_retrain_every: int = BT_ML_RETRAIN_EVERY
    base_seed: int = BT_BASE_SEED
    min_ml_history: int = BT_MIN_ML_HISTORY

    # Dashboard overrides — None means use defaults
    factor_weights: dict | None = None
    opt_max_sector_dev: float | None = None
    opt_max_turnover: float | None = None
    opt_max_single_name: float | None = None
    opt_quality_floor: float | None = None
    ml_n_estimators: int | None = None
    ml_max_depth: int | None = None
    ml_learning_rate: float | None = None
    progress_callback: object = None  # callable(period, n_periods)


@dataclass
class MLBacktestResult:
    # Monthly return series (indexed by period date)
    monthly_returns_ml: pd.Series
    monthly_returns_fixed: pd.Series
    monthly_returns_benchmark: pd.Series

    # Cumulative
    cumulative_ml: pd.Series
    cumulative_fixed: pd.Series
    cumulative_benchmark: pd.Series

    # Weights history
    weights_history_ml: pd.DataFrame
    weights_history_fixed: pd.DataFrame

    # SHAP weights at each rebalance
    shap_weights_history: pd.DataFrame

    # Factor attribution per period
    factor_attribution: pd.DataFrame

    # Turnover
    turnover_ml: pd.Series
    turnover_fixed: pd.Series

    # Summary stats (from _compute_stats)
    stats_ml: dict
    stats_fixed: dict

    # Regime (optional)
    regime_history: Optional[pd.Series] = None


def _compute_stats(strat: pd.Series, bench: pd.Series,
                   turnover: pd.Series) -> dict:
    """Compute summary statistics (replicates bucket_backtest._compute_stats)."""
    excess = strat - bench
    n_months = len(strat)
    if n_months < 2:
        return {"n_months": n_months}

    n_years = n_months / 12.0

    cum_strat = (1 + strat).prod()
    cum_bench = (1 + bench).prod()
    ann_ret_strat = cum_strat ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_ret_bench = cum_bench ** (1 / n_years) - 1 if n_years > 0 else 0

    ann_vol_strat = strat.std() * np.sqrt(12)
    ann_vol_bench = bench.std() * np.sqrt(12)

    sharpe_strat = ann_ret_strat / ann_vol_strat if ann_vol_strat > 0 else 0
    sharpe_bench = ann_ret_bench / ann_vol_bench if ann_vol_bench > 0 else 0

    te = excess.std() * np.sqrt(12)
    ann_alpha = ann_ret_strat - ann_ret_bench
    ir = ann_alpha / te if te > 0 else 0

    t_stat = (excess.mean() / (excess.std() / np.sqrt(n_months))
              if excess.std() > 0 else 0)

    cum = (1 + strat).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    hit_rate = (excess > 0).mean()
    avg_turnover = turnover.mean()

    return {
        "period": f"{strat.index[0].strftime('%Y-%m')} to {strat.index[-1].strftime('%Y-%m')}",
        "n_months": n_months,
        "ann_return_strategy": ann_ret_strat,
        "ann_return_benchmark": ann_ret_bench,
        "ann_alpha": ann_alpha,
        "ann_vol_strategy": ann_vol_strat,
        "ann_vol_benchmark": ann_vol_bench,
        "sharpe_strategy": sharpe_strat,
        "sharpe_benchmark": sharpe_bench,
        "information_ratio": ir,
        "t_stat_alpha": t_stat,
        "max_drawdown": max_dd,
        "hit_rate": hit_rate,
        "avg_monthly_turnover": avg_turnover,
        "tracking_error": te,
    }


def _compute_factor_attribution(
    weights: np.ndarray,
    df: pd.DataFrame,
    fwd_returns: np.ndarray,
    factor_cols: list,
) -> dict:
    """
    Decompose portfolio return by factor.

    For each factor, compute the weighted-average factor exposure times
    the cross-sectional factor return (top quintile - bottom quintile).
    """
    port_return = float(weights @ fwd_returns)
    attribution = {"total": port_return}

    for col in factor_cols:
        if col not in df.columns:
            continue
        scores = df[col].values
        # Portfolio's weighted factor exposure
        exposure = float(weights @ scores)
        # Cross-sectional factor return: correlation of factor with returns
        if np.std(scores) > 1e-10 and np.std(fwd_returns) > 1e-10:
            factor_ret = np.corrcoef(scores, fwd_returns)[0, 1] * np.std(fwd_returns)
        else:
            factor_ret = 0.0
        attribution[col] = exposure * factor_ret

    return attribution


def run_ml_backtest(config: MLBacktestConfig | None = None) -> MLBacktestResult:
    """
    Run the rolling walk-forward ML backtest.

    At each period t:
    1. Build universe (seed varies per period)
    2. Compute forward returns (the realized returns for this period)
    3. If enough history: retrain ML, get SHAP weights
    4. Optimize ML-weighted and fixed-weighted portfolios
    5. Compute portfolio returns net of transaction costs
    """
    if config is None:
        config = MLBacktestConfig()

    # Storage
    dates = []
    rets_ml, rets_fixed, rets_bench = [], [], []
    w_ml_rows, w_fixed_rows = [], []
    shap_rows = []
    attrib_rows = []
    to_ml, to_fixed = [], []

    # State
    prev_w_ml = None
    prev_w_fixed = None
    current_shap_weights = None
    current_model = None
    historical_panels = []

    avail_features = None  # set on first universe

    logger.info("Backtest: %d months, %d bonds, model=%s, TC=%.0fbp",
                config.n_periods, config.n_bonds, config.model_type, config.tc_bps)
    logger.info("ML retrain every %d months", config.ml_retrain_every)

    for t in range(config.n_periods):
        date = pd.Timestamp("2020-01-01") + pd.DateOffset(months=t)
        seed = config.base_seed + t * 7

        # Apply stress shocks in later periods
        shock = None
        if t >= config.n_periods * 3 // 4:
            rng = np.random.default_rng(seed + 999)
            shock = {
                f"BOND{i:03d}": {"spread_6m_chg": float(rng.normal(0, 8))}
                for i in range(0, config.n_bonds, 8)
            }

        # Build universe for this period
        df_t = build_universe(n=config.n_bonds, shock=shock, seed=seed)
        fwd_rets = compute_forward_returns(df_t).values

        # Set available features on first pass
        if avail_features is None:
            avail_features = [f for f in ML_FEATURES if f in df_t.columns]
            opt_factors = [f for f in OPT_FACTOR_WEIGHTS if f in df_t.columns]

        # Add to historical panel for ML training
        df_t_panel = df_t.copy()
        df_t_panel["fwd_excess_return"] = fwd_rets
        df_t_panel["date"] = date
        historical_panels.append(df_t_panel)

        # ── ML retraining ────────────────────────────────────
        need_retrain = (
            t >= config.min_ml_history
            and (current_model is None or t % config.ml_retrain_every == 0)
        )

        if need_retrain:
            # Build panel from all PRIOR periods (no look-ahead)
            panel = pd.concat(historical_panels[:-1], ignore_index=True)

            valid = panel[avail_features + ["fwd_excess_return"]].dropna()
            X_train = valid[avail_features].values
            y_train = valid["fwd_excess_return"].values

            if len(X_train) >= 20:
                current_model = build_model(
                    config.model_type,
                    n_estimators=config.ml_n_estimators,
                    max_depth=config.ml_max_depth,
                    learning_rate=config.ml_learning_rate,
                )
                if config.model_type == "enhanced_rf":
                    X_fit, y_fit = _oversample_extremes(X_train, y_train)
                else:
                    X_fit, y_fit = X_train, y_train
                current_model.fit(X_fit, y_fit)

                # SHAP on current universe to get factor weights
                explainer = shap.TreeExplainer(current_model)
                shap_vals = explainer.shap_values(df_t[avail_features].values)
                current_shap_weights = compute_shap_factor_weights(
                    shap_vals, avail_features
                )

        # ── Build optimizer config overrides ──────────────────
        opt_overrides = {}
        if config.opt_max_sector_dev is not None:
            opt_overrides["max_sector_dev"] = config.opt_max_sector_dev
        if config.opt_max_turnover is not None:
            opt_overrides["max_turnover"] = config.opt_max_turnover
        if config.opt_max_single_name is not None:
            opt_overrides["max_single_name"] = config.opt_max_single_name
        if config.opt_quality_floor is not None:
            opt_overrides["quality_floor"] = config.opt_quality_floor

        # ── SHAP weights for optimizer ───────────────────────
        if current_shap_weights is not None:
            # Map ML features to optimizer factors (redistribute carry weight)
            raw = {f: current_shap_weights.get(f, 0.0) for f in opt_factors}
            total = sum(raw.values())
            if total > 1e-10:
                ml_factor_weights = {f: v / total for f, v in raw.items()}
            else:
                ml_factor_weights = dict(OPT_FACTOR_WEIGHTS)
            config_ml = OptConfig(factor_weights=ml_factor_weights, **opt_overrides)
        else:
            ml_factor_weights = dict(OPT_FACTOR_WEIGHTS)
            config_ml = OptConfig(**opt_overrides)

        shap_row = dict(ml_factor_weights)
        shap_row["date"] = date
        shap_rows.append(shap_row)

        # ── Optimize 3 strategies ────────────────────────────
        # ML-weighted
        r_ml = optimise(df_t, prior_w=prev_w_ml, config=config_ml)
        w_ml = r_ml.weights.values

        # Fixed-weighted: use config.factor_weights if provided
        fixed_fw = config.factor_weights if config.factor_weights else dict(OPT_FACTOR_WEIGHTS)
        r_fixed = optimise(df_t, prior_w=prev_w_fixed,
                           config=OptConfig(factor_weights=fixed_fw, **opt_overrides))
        w_fixed = r_fixed.weights.values

        # Benchmark: equal weight
        n = len(df_t[df_t["quality_score"] >= config_ml.quality_floor])
        w_bench = np.ones(n) / n

        # Align weights to same eligible bonds
        eligible = df_t["quality_score"] >= config_ml.quality_floor
        fwd_eligible = fwd_rets[eligible.values]

        # ── Compute returns ──────────────────────────────────
        # Turnover
        if prev_w_ml is not None and len(prev_w_ml) == len(w_ml):
            to_ml_val = np.sum(np.abs(w_ml - prev_w_ml)) / 2.0
        else:
            to_ml_val = 0.0

        if prev_w_fixed is not None and len(prev_w_fixed) == len(w_fixed):
            to_fixed_val = np.sum(np.abs(w_fixed - prev_w_fixed)) / 2.0
        else:
            to_fixed_val = 0.0

        tc_ml = to_ml_val * config.tc_bps / 10000.0
        tc_fixed = to_fixed_val * config.tc_bps / 10000.0

        ret_ml = float(w_ml @ fwd_eligible) - tc_ml
        ret_fixed = float(w_fixed @ fwd_eligible) - tc_fixed
        ret_bench = float(w_bench @ fwd_eligible)

        # ── Factor attribution ───────────────────────────────
        df_eligible = df_t[eligible].reset_index(drop=True)
        attrib = _compute_factor_attribution(w_ml, df_eligible, fwd_eligible, opt_factors)
        attrib["date"] = date
        attrib_rows.append(attrib)

        # ── Store ────────────────────────────────────────────
        dates.append(date)
        rets_ml.append(ret_ml)
        rets_fixed.append(ret_fixed)
        rets_bench.append(ret_bench)
        w_ml_rows.append(w_ml)
        w_fixed_rows.append(w_fixed)
        to_ml.append(to_ml_val)
        to_fixed.append(to_fixed_val)

        prev_w_ml = w_ml
        prev_w_fixed = w_fixed

        # Progress callback for dashboard
        if config.progress_callback is not None:
            config.progress_callback(t + 1, config.n_periods)

        # Progress
        if (t + 1) % 12 == 0 or t == config.n_periods - 1:
            cum_ml = (1 + pd.Series(rets_ml)).prod() - 1
            cum_fixed = (1 + pd.Series(rets_fixed)).prod() - 1
            logger.info("Month %3d: ML cum=%+.2f%%  Fixed cum=%+.2f%%  SHAP: %s",
                       t + 1, cum_ml * 100, cum_fixed * 100,
                       _format_shap(ml_factor_weights))

    # ── Build results ────────────────────────────────────────
    idx = pd.DatetimeIndex(dates)

    s_ml = pd.Series(rets_ml, index=idx, name="ml")
    s_fixed = pd.Series(rets_fixed, index=idx, name="fixed")
    s_bench = pd.Series(rets_bench, index=idx, name="benchmark")

    stats_ml = _compute_stats(s_ml, s_bench, pd.Series(to_ml, index=idx))
    stats_fixed = _compute_stats(s_fixed, s_bench, pd.Series(to_fixed, index=idx))

    # SHAP weights history
    shap_df = pd.DataFrame(shap_rows).set_index("date")
    shap_df.index = pd.to_datetime(shap_df.index)

    # Factor attribution
    attrib_df = pd.DataFrame(attrib_rows).set_index("date")
    attrib_df.index = pd.to_datetime(attrib_df.index)

    return MLBacktestResult(
        monthly_returns_ml=s_ml,
        monthly_returns_fixed=s_fixed,
        monthly_returns_benchmark=s_bench,
        cumulative_ml=(1 + s_ml).cumprod(),
        cumulative_fixed=(1 + s_fixed).cumprod(),
        cumulative_benchmark=(1 + s_bench).cumprod(),
        weights_history_ml=pd.DataFrame(w_ml_rows, index=idx),
        weights_history_fixed=pd.DataFrame(w_fixed_rows, index=idx),
        shap_weights_history=shap_df,
        factor_attribution=attrib_df,
        turnover_ml=pd.Series(to_ml, index=idx, name="turnover_ml"),
        turnover_fixed=pd.Series(to_fixed, index=idx, name="turnover_fixed"),
        stats_ml=stats_ml,
        stats_fixed=stats_fixed,
    )


def format_stats_table(result: MLBacktestResult) -> str:
    """Format three-strategy comparison table."""
    ml = result.stats_ml
    fx = result.stats_fixed
    lines = []
    lines.append("=" * 70)
    lines.append("  ML FACTOR BACKTEST — Three-Strategy Comparison")
    lines.append(f"  {ml.get('period', 'N/A')}  ({ml.get('n_months', 0)} months)")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"  {'':30s}  {'ML-Weighted':>12s}  {'Fixed-Weight':>12s}")
    lines.append(f"  {'-'*56}")
    lines.append(f"  {'Annualised Return':30s}  {ml['ann_return_strategy']:>+11.2%}  {fx['ann_return_strategy']:>+11.2%}")
    lines.append(f"  {'Annualised Volatility':30s}  {ml['ann_vol_strategy']:>11.2%}  {fx['ann_vol_strategy']:>11.2%}")
    lines.append(f"  {'Sharpe Ratio':30s}  {ml['sharpe_strategy']:>11.2f}  {fx['sharpe_strategy']:>11.2f}")
    lines.append(f"  {'Max Drawdown':30s}  {ml['max_drawdown']:>11.2%}  {fx['max_drawdown']:>11.2%}")
    lines.append("")
    lines.append(f"  {'Alpha vs Benchmark':30s}  {ml['ann_alpha']:>+11.2%}  {fx['ann_alpha']:>+11.2%}")
    lines.append(f"  {'Tracking Error':30s}  {ml['tracking_error']:>11.2%}  {fx['tracking_error']:>11.2%}")
    lines.append(f"  {'Information Ratio':30s}  {ml['information_ratio']:>11.2f}  {fx['information_ratio']:>11.2f}")
    lines.append(f"  {'t-stat (alpha)':30s}  {ml['t_stat_alpha']:>11.2f}  {fx['t_stat_alpha']:>11.2f}")
    lines.append(f"  {'Hit Rate':30s}  {ml['hit_rate']:>11.1%}  {fx['hit_rate']:>11.1%}")
    lines.append(f"  {'Avg Monthly Turnover':30s}  {ml['avg_monthly_turnover']:>11.2%}  {fx['avg_monthly_turnover']:>11.2%}")
    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


def _format_shap(w: dict) -> str:
    """One-line SHAP weight summary."""
    parts = [f"{k.replace('z_','')[:3]}={v:.0%}" for k, v in
             sorted(w.items(), key=lambda x: -x[1])]
    return " ".join(parts)
