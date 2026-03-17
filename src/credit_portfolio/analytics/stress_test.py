"""
Stress-testing framework for credit portfolio — bucket level.

Operates on the 4 IG rating buckets (AAA, AA, A, BBB) using real FRED
OAS data.  Applies predefined or custom OAS shocks, computes the impact
on portfolio weights via signal-tilted rebalancing, and compares
baseline vs stressed allocations.

Predefined scenarios:
  1. Spread Widening +200bp  — uniform shock across all buckets
  2. BBB Crisis              — BBB +300bp, A +50bp
  3. Fed Hike Shock          — all +100bp
  4. COVID Replay            — all +280bp
  5. Custom                  — user-specified per-rating shocks
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd

from credit_portfolio.backtests.bucket_backtest import (
    RATING_BUCKETS, OAS_COLS, DURATIONS_BY_BUCKET, MKT_WEIGHTS,
    BacktestConfig, _compute_signals, _composite_signal, _tilt_weights,
    _compute_monthly_excess_returns,
)


# ── Predefined Scenarios ────────────────────────────────────────────────────

SCENARIOS = {
    "Spread Widening +200bp": {
        "description": "Uniform +200bp OAS shock to all rating buckets",
        "shocks": {"AAA": 2.00, "AA": 2.00, "A": 2.00, "BBB": 2.00},
    },
    "BBB Crisis": {
        "description": "BBB spreads blow out +300bp, A widens +50bp",
        "shocks": {"AAA": 0.0, "AA": 0.0, "A": 0.50, "BBB": 3.00},
    },
    "Fed Hike Shock": {
        "description": "All buckets +100bp",
        "shocks": {"AAA": 1.00, "AA": 1.00, "A": 1.00, "BBB": 1.00},
    },
    "COVID Replay": {
        "description": "All buckets +280bp",
        "shocks": {"AAA": 2.80, "AA": 2.80, "A": 2.80, "BBB": 2.80},
    },
}


@dataclass
class StressResult:
    """Output of a bucket-level stress test."""
    scenario_name: str
    description: str
    # Bucket-level data
    baseline_oas: dict          # {bucket: oas_pct}  (current real OAS)
    stressed_oas: dict          # {bucket: oas_pct}  (after shock)
    shocks_applied: dict        # {bucket: delta_pct}
    # Weights
    baseline_weights: np.ndarray   # shape (4,) — signal-tilted baseline
    stressed_weights: np.ndarray   # shape (4,) — signal-tilted after shock
    market_weights: np.ndarray     # shape (4,) — benchmark
    weight_changes: pd.DataFrame   # per-bucket comparison table
    # Return impact
    baseline_carry: dict        # monthly carry per bucket (baseline)
    stressed_carry: dict        # monthly carry per bucket (stressed)
    price_impact: dict          # price return from OAS shock per bucket


def run_stress_test(
    monthly: pd.DataFrame,
    scenario: str = "Spread Widening +200bp",
    config: BacktestConfig | None = None,
    custom_shocks: dict | None = None,
) -> StressResult:
    """
    Run a bucket-level stress test on real FRED OAS data.

    Parameters
    ----------
    monthly : DataFrame from loader.load() with OAS columns.
    scenario : Name of predefined scenario, or "Custom".
    config : BacktestConfig for signal-tilted weight computation.
    custom_shocks : {rating: delta_oas_pct} for Custom scenario.
                    OAS shocks in percentage points (e.g., 2.0 = +200bp).

    Returns
    -------
    StressResult with baseline vs stressed bucket allocations.
    """
    if config is None:
        config = BacktestConfig()

    # Determine shocks
    if scenario == "Custom":
        shocks = custom_shocks or {}
        description = "Custom user-defined shocks"
    else:
        sc = SCENARIOS.get(scenario, SCENARIOS["Spread Widening +200bp"])
        shocks = sc["shocks"]
        description = sc["description"]

    # Get current (latest) OAS per bucket from real data
    baseline_oas = {}
    for bucket in RATING_BUCKETS:
        col = OAS_COLS[bucket]
        if col in monthly.columns:
            val = monthly[col].dropna().iloc[-1]
            baseline_oas[bucket] = float(val)

    # Apply shocks
    stressed_oas = {}
    for bucket in RATING_BUCKETS:
        stressed_oas[bucket] = baseline_oas.get(bucket, 0.0) + shocks.get(bucket, 0.0)

    # Build stressed monthly DataFrame (append a shocked row)
    stressed_monthly = monthly.copy()
    last_date = stressed_monthly.index[-1]
    shock_date = last_date + pd.DateOffset(months=1)
    shock_row = stressed_monthly.iloc[-1].copy()
    for bucket in RATING_BUCKETS:
        col = OAS_COLS[bucket]
        if col in shock_row.index:
            shock_row[col] = stressed_oas[bucket]
    stressed_monthly.loc[shock_date] = shock_row

    # Compute excess returns for signal computation
    excess_returns_base = _compute_monthly_excess_returns(monthly)
    excess_returns_stressed = _compute_monthly_excess_returns(stressed_monthly)

    # Compute baseline weights (signals at latest available date)
    t_base = len(excess_returns_base) - 1
    if t_base >= config.momentum_window:
        signals_base = _compute_signals(monthly, excess_returns_base, t_base, config)
        composite_base = _composite_signal(signals_base, config.factor_weights)
        baseline_weights = _tilt_weights(composite_base, config)
    else:
        baseline_weights = MKT_WEIGHTS.copy()

    # Compute stressed weights (signals at the shocked date)
    t_stress = len(excess_returns_stressed) - 1
    if t_stress >= config.momentum_window:
        signals_stress = _compute_signals(stressed_monthly, excess_returns_stressed, t_stress, config)
        composite_stress = _composite_signal(signals_stress, config.factor_weights)
        stressed_weights = _tilt_weights(composite_stress, config)
    else:
        stressed_weights = MKT_WEIGHTS.copy()

    # Build comparison table
    weight_changes = pd.DataFrame({
        "Bucket": RATING_BUCKETS,
        "Baseline OAS (%)": [baseline_oas.get(b, 0.0) for b in RATING_BUCKETS],
        "Stressed OAS (%)": [stressed_oas.get(b, 0.0) for b in RATING_BUCKETS],
        "Shock (bp)": [shocks.get(b, 0.0) * 100 for b in RATING_BUCKETS],
        "Market Weight": MKT_WEIGHTS,
        "Baseline Weight": baseline_weights,
        "Stressed Weight": stressed_weights,
        "Weight Change": stressed_weights - baseline_weights,
    })

    # Carry and price impact
    baseline_carry = {}
    stressed_carry = {}
    price_impact = {}
    for bucket in RATING_BUCKETS:
        dur = DURATIONS_BY_BUCKET[bucket]
        base_oas = baseline_oas.get(bucket, 0.0)
        stress_oas = stressed_oas.get(bucket, 0.0)
        delta = shocks.get(bucket, 0.0)

        baseline_carry[bucket] = base_oas / 100.0 / 12.0      # monthly carry
        stressed_carry[bucket] = stress_oas / 100.0 / 12.0
        price_impact[bucket] = -dur * delta / 100.0            # price return from shock

    return StressResult(
        scenario_name=scenario,
        description=description,
        baseline_oas=baseline_oas,
        stressed_oas=stressed_oas,
        shocks_applied=shocks,
        baseline_weights=baseline_weights,
        stressed_weights=stressed_weights,
        market_weights=MKT_WEIGHTS,
        weight_changes=weight_changes,
        baseline_carry=baseline_carry,
        stressed_carry=stressed_carry,
        price_impact=price_impact,
    )
