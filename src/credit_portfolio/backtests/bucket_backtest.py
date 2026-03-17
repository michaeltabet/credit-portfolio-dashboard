"""
Bucket-level credit factor rotation backtest.

Methodology
-----------
Universe : ICE BofA IG rating-bucket indices (AAA, AA, A, BBB) sourced from FRED.
Frequency: Monthly rebalance on month-end.
Signals  : Three credit factors computed at each rebalance date using only
           information available at that point (no look-ahead):
             1. DTS  — Duration × Spread (OAS × effective duration)
             2. Value — Cross-sectional z-score of log(OAS) across buckets
             3. Momentum — Trailing 6-month excess return

Returns  : Monthly excess return per bucket estimated as:
             r_i = carry + price_return
             carry       = OAS_i / 12
             price_return = −D_i × ΔOAS_i
           where OAS is in decimal (0.82% = 0.0082) and D is effective
           duration in years.  This follows the standard spread-duration
           approximation used in Barclays, JP Morgan, and ICE index analytics.

Benchmark: Market-capitalisation weights (AAA 3%, AA 10%, A 38%, BBB 49%).
Strategy : Signal-tilted portfolio.  At each rebalance the composite z-score
           is used to shift weight toward high-signal buckets:
             w_i = w_mkt_i + tilt_strength × z_composite_i / Σ|z|
           Weights are floored at 1% and renormalised to sum to 1.

Transaction costs: One-way cost applied to absolute weight turnover.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from credit_portfolio.config import load_config


def _load_defaults() -> dict:
    """Load defaults from config.yaml at import time."""
    cfg = load_config()
    return cfg


_CFG = _load_defaults()

# All of these are read from config.yaml — no hardcoded values
_universe = _CFG.get("universe", {})
_factors = _CFG.get("factors", {})
_bt = _CFG.get("backtest", {})

RATING_BUCKETS = _universe.get("ratings", ["AAA", "AA", "A", "BBB"])
OAS_COLS = _universe.get("rating_oas_col", {
    "AAA": "oas_aaa", "AA": "oas_aa", "A": "oas_a", "BBB": "oas_bbb",
})
DURATIONS_BY_BUCKET = _universe.get("durations", {
    "AAA": 8.5, "AA": 7.5, "A": 7.2, "BBB": 6.5,
})
_mkt_w = _universe.get("market_weights", {
    "AAA": 0.04, "AA": 0.12, "A": 0.34, "BBB": 0.50,
})
MKT_WEIGHTS = np.array([_mkt_w.get(b, 0.25) for b in RATING_BUCKETS])


@dataclass
class BacktestConfig:
    tilt_strength: float = _bt.get("tilt_strength", 0.10)
    tc_bps: float = _bt.get("transaction_cost_bps", 5.0)
    momentum_window: int = _factors.get("momentum_window", 6)
    min_weight: float = _bt.get("min_weight", 0.01)
    factor_weights: dict = field(default_factory=lambda: dict(
        _factors.get("weights", {"z_dts": 0.50, "z_value": 0.25, "z_momentum": 0.25})
    ))


@dataclass
class BacktestResult:
    # Time series (indexed by month-end date)
    monthly_returns_strategy: pd.Series
    monthly_returns_benchmark: pd.Series
    cumulative_strategy: pd.Series
    cumulative_benchmark: pd.Series
    weights_history: pd.DataFrame        # columns = rating buckets
    signals_history: pd.DataFrame        # columns = factor signals per bucket
    turnover: pd.Series

    # Summary statistics
    stats: dict


# ── Return computation ───────────────────────────────────────────────────────

def _compute_monthly_excess_returns(monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Compute monthly excess return per rating bucket from OAS data.

    r_i(t) = carry(t) + price_return(t)
    carry       = OAS(t) / 12
    price_return = −duration × [OAS(t+1) − OAS(t)]

    OAS from FRED is in % (e.g., 0.82 means 82bp → 0.0082 in decimal).
    Returns are in decimal (0.01 = 1%).
    """
    returns = pd.DataFrame(index=monthly.index)
    for bucket in RATING_BUCKETS:
        col = OAS_COLS[bucket]
        if col not in monthly.columns:
            continue
        oas_pct = monthly[col]                       # e.g., 0.82 for 82bp
        dur = DURATIONS_BY_BUCKET[bucket]

        carry = oas_pct / 100.0 / 12.0               # monthly carry in decimal
        delta_oas = oas_pct.diff(1) / 100.0           # change in OAS (decimal)
        price_ret = -dur * delta_oas                   # price return from spread move

        returns[bucket] = carry + price_ret

    return returns.iloc[1:]  # drop first row (NaN from diff)


# ── Signal computation ───────────────────────────────────────────────────────

def _compute_signals(monthly: pd.DataFrame, excess_returns: pd.DataFrame,
                     t: int, config: BacktestConfig) -> dict:
    """
    Compute factor signals at rebalance date index t.
    Uses only data available at time t (no look-ahead).
    Returns dict of {signal_name: {bucket: z-score}}.
    """
    signals = {}

    # --- DTS: duration × spread level ---
    dts_raw = {}
    for bucket in RATING_BUCKETS:
        col = OAS_COLS[bucket]
        if col in monthly.columns:
            oas = monthly[col].iloc[t]
            if pd.notna(oas):
                dts_raw[bucket] = DURATIONS_BY_BUCKET[bucket] * oas
    if dts_raw:
        vals = np.array(list(dts_raw.values()))
        mu, sd = vals.mean(), vals.std()
        if sd > 1e-10:
            signals["z_dts"] = {b: (v - mu) / sd for b, v in dts_raw.items()}
        else:
            signals["z_dts"] = {b: 0.0 for b in dts_raw}

    # --- Value: cross-sectional z-score of log(OAS) ---
    log_oas = {}
    for bucket in RATING_BUCKETS:
        col = OAS_COLS[bucket]
        if col in monthly.columns:
            oas = monthly[col].iloc[t]
            if pd.notna(oas) and oas > 0:
                log_oas[bucket] = np.log(oas)
    if log_oas:
        vals = np.array(list(log_oas.values()))
        mu, sd = vals.mean(), vals.std()
        if sd > 1e-10:
            signals["z_value"] = {b: (v - mu) / sd for b, v in log_oas.items()}
        else:
            signals["z_value"] = {b: 0.0 for b in log_oas}

    # --- Momentum: trailing N-month cumulative excess return ---
    mom_window = config.momentum_window
    if t >= mom_window:
        trailing = excess_returns.iloc[t - mom_window:t]
        cum_ret = (1 + trailing).prod() - 1
        vals = cum_ret.values
        mu, sd = vals.mean(), vals.std()
        if sd > 1e-10:
            signals["z_momentum"] = {b: (cum_ret[b] - mu) / sd for b in RATING_BUCKETS}
        else:
            signals["z_momentum"] = {b: 0.0 for b in RATING_BUCKETS}

    return signals


def _composite_signal(signals: dict, factor_weights: dict) -> dict:
    """Weighted-average composite score per bucket."""
    composite = {b: 0.0 for b in RATING_BUCKETS}
    total_weight = 0.0
    for sig_name, w in factor_weights.items():
        if sig_name in signals:
            total_weight += w
            for b in RATING_BUCKETS:
                composite[b] += w * signals[sig_name].get(b, 0.0)
    if total_weight > 0:
        composite = {b: v / total_weight for b, v in composite.items()}
    return composite


# ── Portfolio construction ───────────────────────────────────────────────────

def _tilt_weights(composite: dict, config: BacktestConfig) -> np.ndarray:
    """
    Tilt market weights based on composite signal.

    w_i = w_mkt_i + tilt_strength × z_i / sum(|z|)
    Then floor at min_weight and renormalise.
    """
    z = np.array([composite[b] for b in RATING_BUCKETS])
    abs_sum = np.abs(z).sum()
    if abs_sum < 1e-10:
        return MKT_WEIGHTS.copy()

    tilt = config.tilt_strength * z / abs_sum
    w = MKT_WEIGHTS + tilt
    w = np.maximum(w, config.min_weight)
    w /= w.sum()
    return w


# ── Main backtest loop ───────────────────────────────────────────────────────

def run_backtest(monthly: pd.DataFrame,
                 config: BacktestConfig | None = None) -> BacktestResult:
    """
    Run the bucket rotation backtest.

    Parameters
    ----------
    monthly : DataFrame with month-end OAS columns (oas_aaa, oas_aa, etc.)
              as loaded by loader.load().
    config  : Backtest configuration. Defaults to sensible values.

    Returns
    -------
    BacktestResult with full time series and summary statistics.
    """
    if config is None:
        config = BacktestConfig()

    # Compute monthly excess returns for each bucket
    excess_returns = _compute_monthly_excess_returns(monthly)

    # Align indices
    common_idx = excess_returns.dropna().index
    excess_returns = excess_returns.loc[common_idx]

    n_months = len(excess_returns)
    start_idx = config.momentum_window + 1  # need history for momentum

    # Storage
    strat_rets = []
    bench_rets = []
    weights_rows = []
    signal_rows = []
    turnover_list = []
    dates = []

    prev_weights = MKT_WEIGHTS.copy()

    for t in range(start_idx, n_months):
        date = excess_returns.index[t]

        # Compute signals at t-1 (rebalance before observing month t return)
        signals = _compute_signals(monthly, excess_returns, t - 1, config)
        composite = _composite_signal(signals, config.factor_weights)

        # Target weights
        target_w = _tilt_weights(composite, config)

        # Turnover (one-way)
        turnover = np.sum(np.abs(target_w - prev_weights)) / 2.0

        # Transaction cost drag
        tc_drag = turnover * config.tc_bps / 10000.0

        # Monthly returns
        month_ret = excess_returns.iloc[t].values
        strat_ret = float(np.dot(target_w, month_ret)) - tc_drag
        bench_ret = float(np.dot(MKT_WEIGHTS, month_ret))

        strat_rets.append(strat_ret)
        bench_rets.append(bench_ret)
        weights_rows.append(target_w.copy())
        turnover_list.append(turnover)
        dates.append(date)

        # Flatten signals for history
        sig_row = {}
        for sig_name, bucket_scores in signals.items():
            for b, v in bucket_scores.items():
                sig_row[f"{sig_name}_{b}"] = v
        sig_row.update({f"composite_{b}": v for b, v in composite.items()})
        signal_rows.append(sig_row)

        prev_weights = target_w

    # Build output series
    idx = pd.DatetimeIndex(dates)

    strat_series = pd.Series(strat_rets, index=idx, name="strategy")
    bench_series = pd.Series(bench_rets, index=idx, name="benchmark")
    cum_strat = (1 + strat_series).cumprod()
    cum_bench = (1 + bench_series).cumprod()

    weights_df = pd.DataFrame(weights_rows, index=idx, columns=RATING_BUCKETS)
    signals_df = pd.DataFrame(signal_rows, index=idx)
    turnover_series = pd.Series(turnover_list, index=idx, name="turnover")

    # Summary statistics
    stats = _compute_stats(strat_series, bench_series, turnover_series)

    return BacktestResult(
        monthly_returns_strategy=strat_series,
        monthly_returns_benchmark=bench_series,
        cumulative_strategy=cum_strat,
        cumulative_benchmark=cum_bench,
        weights_history=weights_df,
        signals_history=signals_df,
        turnover=turnover_series,
        stats=stats,
    )


# ── Performance statistics ───────────────────────────────────────────────────

def _compute_stats(strat: pd.Series, bench: pd.Series,
                   turnover: pd.Series) -> dict:
    """Compute summary statistics for the backtest."""
    excess = strat - bench
    n_months = len(strat)
    n_years = n_months / 12.0

    # Annualised return
    cum_strat = (1 + strat).prod()
    cum_bench = (1 + bench).prod()
    ann_ret_strat = cum_strat ** (1 / n_years) - 1
    ann_ret_bench = cum_bench ** (1 / n_years) - 1

    # Annualised volatility
    ann_vol_strat = strat.std() * np.sqrt(12)
    ann_vol_bench = bench.std() * np.sqrt(12)

    # Sharpe ratio (using excess returns over benchmark as risk-free proxy = 0)
    sharpe_strat = ann_ret_strat / ann_vol_strat if ann_vol_strat > 0 else 0
    sharpe_bench = ann_ret_bench / ann_vol_bench if ann_vol_bench > 0 else 0

    # Information ratio
    te = excess.std() * np.sqrt(12)  # tracking error (annualised)
    ann_alpha = ann_ret_strat - ann_ret_bench
    ir = ann_alpha / te if te > 0 else 0

    # t-statistic on alpha
    t_stat = excess.mean() / (excess.std() / np.sqrt(n_months)) if excess.std() > 0 else 0

    # Max drawdown
    cum = (1 + strat).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    cum_b = (1 + bench).cumprod()
    peak_b = cum_b.cummax()
    dd_b = (cum_b - peak_b) / peak_b
    max_dd_bench = dd_b.min()

    # Hit rate (months where strategy outperforms)
    hit_rate = (excess > 0).mean()

    # Average turnover
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
        "max_drawdown_strategy": max_dd,
        "max_drawdown_benchmark": max_dd_bench,
        "hit_rate": hit_rate,
        "avg_monthly_turnover": avg_turnover,
        "tracking_error": te,
    }


def format_stats_table(stats: dict) -> str:
    """Format summary statistics as a publication-ready table."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"  CREDIT FACTOR ROTATION BACKTEST")
    lines.append(f"  {stats['period']}  ({stats['n_months']} months)")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"  {'':30s}  {'Strategy':>10s}  {'Benchmark':>10s}")
    lines.append(f"  {'-'*52}")
    lines.append(f"  {'Annualised Return':30s}  {stats['ann_return_strategy']:>+9.2%}  {stats['ann_return_benchmark']:>+9.2%}")
    lines.append(f"  {'Annualised Volatility':30s}  {stats['ann_vol_strategy']:>9.2%}  {stats['ann_vol_benchmark']:>9.2%}")
    lines.append(f"  {'Sharpe Ratio':30s}  {stats['sharpe_strategy']:>9.2f}  {stats['sharpe_benchmark']:>9.2f}")
    lines.append(f"  {'Max Drawdown':30s}  {stats['max_drawdown_strategy']:>9.2%}  {stats['max_drawdown_benchmark']:>9.2%}")
    lines.append("")
    lines.append(f"  {'Annualised Alpha':30s}  {stats['ann_alpha']:>+9.2%}")
    lines.append(f"  {'Tracking Error':30s}  {stats['tracking_error']:>9.2%}")
    lines.append(f"  {'Information Ratio':30s}  {stats['information_ratio']:>9.2f}")
    lines.append(f"  {'t-stat (alpha)':30s}  {stats['t_stat_alpha']:>9.2f}")
    lines.append(f"  {'Hit Rate':30s}  {stats['hit_rate']:>9.1%}")
    lines.append(f"  {'Avg Monthly Turnover':30s}  {stats['avg_monthly_turnover']:>9.2%}")
    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)
