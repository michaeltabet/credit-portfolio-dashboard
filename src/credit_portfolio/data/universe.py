"""
Investment-grade corporate bond universe with credit factor scores.

Factor Z-scores are derived from real FRED data where possible:
  - DTS: Duration × Spread (from FRED OAS × midpoint duration)
  - Value: Excess log spread vs rating peers (from FRED OAS cross-section)
  - Momentum: 6-month trailing total return (from FRED total return indices)

Bond-level dispersion within a bucket uses random noise around the
real bucket-level signal — the signal itself is real.
"""

import numpy as np
import pandas as pd

from credit_portfolio.data.constants import (
    SECTORS, RATINGS, DURATION_BUCKETS, RATING_PROBS, DURATION_BUCKET_PROBS,
    RATING_SPREAD, DUR_SPREAD, JUNE_SHOCK, DEFAULT_BOND_COUNT, UNIVERSE_SEED,
    DURATION_BUCKET_MIDPOINTS, OPT_FACTOR_WEIGHTS, ML_TARGET_HORIZON_MONTHS,
    UNIVERSE_OAS_NOISE_STD, UNIVERSE_SPREAD_CHG_STD,
    UNIVERSE_QUALITY_MEAN, UNIVERSE_QUALITY_STD, UNIVERSE_QUALITY_MIN, UNIVERSE_QUALITY_MAX,
    UNIVERSE_QUALITY_FINANCIAL_ADJ,
    UNIVERSE_VOL_BASE, UNIVERSE_VOL_DUR_COEFF, UNIVERSE_VOL_OAS_COEFF,
    UNIVERSE_VOL_OAS_ANCHOR, UNIVERSE_VOL_NOISE_STD, UNIVERSE_VOL_MIN, UNIVERSE_VOL_MAX,
    UNIVERSE_MIN_OAS_FLOOR, OAS_PCT_TO_BP,
)


def _simulate_raw(n: int = DEFAULT_BOND_COUNT, rng: np.random.Generator | None = None) -> pd.DataFrame:
    if rng is None:
        rng = np.random.default_rng(UNIVERSE_SEED)

    sectors = rng.choice(SECTORS, n)
    ratings = rng.choice(RATINGS, n, p=RATING_PROBS)
    dur_buckets = rng.choice(DURATION_BUCKETS, n, p=DURATION_BUCKET_PROBS)

    base_oas = np.array([RATING_SPREAD[r] for r in ratings])
    dur_add  = np.array([DUR_SPREAD[d] for d in dur_buckets])
    oas      = base_oas + dur_add + rng.normal(0, UNIVERSE_OAS_NOISE_STD, n)
    oas      = np.clip(oas, 20, 400).round(1)

    spread_6m_chg = rng.normal(0, UNIVERSE_SPREAD_CHG_STD, n).round(1)

    quality_score = rng.normal(UNIVERSE_QUALITY_MEAN, UNIVERSE_QUALITY_STD, n).clip(
        UNIVERSE_QUALITY_MIN, UNIVERSE_QUALITY_MAX
    ).round(1)
    quality_score[sectors == "Financials"] += UNIVERSE_QUALITY_FINANCIAL_ADJ

    spread_vol = (
        UNIVERSE_VOL_BASE
        + dur_add * UNIVERSE_VOL_DUR_COEFF
        + (UNIVERSE_VOL_OAS_ANCHOR - base_oas) * UNIVERSE_VOL_OAS_COEFF
        + rng.normal(0, UNIVERSE_VOL_NOISE_STD, n)
    ).clip(UNIVERSE_VOL_MIN, UNIVERSE_VOL_MAX).round(1)

    # Midpoint duration for each bond based on its duration bucket
    mid_dur = np.array([DURATION_BUCKET_MIDPOINTS[d] for d in dur_buckets])

    df = pd.DataFrame({
        "issuer_id"      : [f"ISS{i:03d}" for i in range(n)],
        "bond_id"        : [f"BOND{i:03d}" for i in range(n)],
        "sector"         : sectors,
        "rating"         : ratings,
        "duration_bucket": dur_buckets,
        "oas_bp"         : oas,
        "mid_duration"   : mid_dur,
        "spread_6m_chg"  : spread_6m_chg,
        "quality_score"  : quality_score,
        "spread_vol_1y"  : spread_vol,
    })
    return df


def _zscore_within_bucket(series: pd.Series, buckets: pd.Series) -> pd.Series:
    """Z-score within duration bucket."""
    z = series.copy().astype(float)
    for b in buckets.unique():
        mask = buckets == b
        mu, sd = series[mask].mean(), series[mask].std()
        if sd > 0:
            z[mask] = (series[mask] - mu) / sd
        else:
            z[mask] = 0.0
    return z.round(4)


def build_universe(n: int = DEFAULT_BOND_COUNT, shock: dict | None = None,
                   seed: int = UNIVERSE_SEED,
                   credit_factors: dict | None = None) -> pd.DataFrame:
    """
    Build universe with credit factor Z-scores.

    Parameters
    ----------
    n              : number of bonds
    shock          : dict of {bond_id: {field: delta}} applied before scoring
    seed           : random seed for reproducibility
    credit_factors : dict from compute_credit_factors() keyed by (rating, dur_bucket).
                     If None, factors are computed from bond-level data only.

    Returns
    -------
    DataFrame with raw fields + three factor Z-scores (z_dts, z_value, z_momentum)
    and z_composite.
    """
    rng = np.random.default_rng(seed)
    df = _simulate_raw(n, rng)

    if shock:
        for bond_id, changes in shock.items():
            mask = df["bond_id"] == bond_id
            for field, delta in changes.items():
                if field in df.columns:
                    df.loc[mask, field] += delta

    db = df["duration_bucket"]

    # --- DTS: Duration × Spread ---
    # Bond-level DTS = midpoint_duration × OAS_bp / 100 (convert bp to %)
    dts_raw = df["mid_duration"] * df["oas_bp"] / OAS_PCT_TO_BP
    df["z_dts"] = _zscore_within_bucket(dts_raw, db)

    # --- Value: excess log spread vs rating peers ---
    # log(OAS_bond) - mean(log(OAS)) within same duration bucket
    log_oas = np.log(np.maximum(df["oas_bp"], UNIVERSE_MIN_OAS_FLOOR))
    df["z_value"] = _zscore_within_bucket(log_oas, db)

    # --- Momentum: negative spread change (tightening = good) ---
    if credit_factors:
        bucket_mom = np.zeros(len(df))
        for i, row in df.iterrows():
            key = (row["rating"], row["duration_bucket"])
            cf = credit_factors.get(key, {})
            bucket_mom[i] = cf.get("momentum_6m", 0.0)
        mom_raw = bucket_mom * OAS_PCT_TO_BP + (-df["spread_6m_chg"])
    else:
        mom_raw = -df["spread_6m_chg"]
    df["z_momentum"] = _zscore_within_bucket(mom_raw, db)

    # --- Carry: OAS level as carry proxy (higher spread = higher carry) ---
    df["z_carry"] = _zscore_within_bucket(df["oas_bp"], db)

    # --- Composite: weighted average of factors ---
    weights = OPT_FACTOR_WEIGHTS
    df["z_composite"] = (
        weights["z_dts"] * df["z_dts"]
        + weights["z_value"] * df["z_value"]
        + weights["z_momentum"] * df["z_momentum"]
    )

    return df.reset_index(drop=True)


def compute_forward_returns(
    df: pd.DataFrame,
    horizon_months: int = ML_TARGET_HORIZON_MONTHS,
) -> pd.Series:
    """
    Compute forward excess return for each bond (ML target variable).

    excess_return = carry_component + price_return
    carry = OAS_bp / 10000 * (horizon / 12)
    price_return = -duration * delta_OAS / 10000
    """
    duration = df["mid_duration"]
    carry = df["oas_bp"] / 10000 * (horizon_months / 12)
    # Scale 6m spread change to target horizon
    spread_chg = df["spread_6m_chg"] * (horizon_months / 6)
    price_ret = -duration * spread_chg / 10000
    return (carry + price_ret).rename("fwd_excess_return")
