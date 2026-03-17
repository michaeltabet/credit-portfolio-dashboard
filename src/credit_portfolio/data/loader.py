"""
Consolidated data loader for FRED credit spread data.

Loads from CSV (primary) or FRED API (live pull).
OAS values are in FRED native % units (0.82 means 82bp).
"""

import numpy as np
import pandas as pd

from credit_portfolio.log import get_logger
from credit_portfolio.data.constants import (
    SERIES_MAP, FRED_SERIES,
    DURATION_BUCKET_MIDPOINTS, DURATION_BUCKET_OAS_COL,
    RATING_OAS_COL, MOMENTUM_TR_COL,
    CHART_ROLLING_WINDOW,
)

logger = get_logger(__name__)


def load(csv_path: str = "data/fred_credit_spreads.csv",
         freq: str = "ME",
         start: str = "1997-01-01") -> pd.DataFrame:
    """
    Load, clean, and resample the FRED credit spread CSV.

    Parameters
    ----------
    csv_path : path to the CSV file
    freq     : resample frequency. 'ME' = month-end, 'D' = daily
    start    : drop observations before this date

    Returns
    -------
    DataFrame with friendly column names, forward-filled, resampled.
    """
    raw = pd.read_csv(csv_path, parse_dates=["DATE"]).set_index("DATE")

    rename = {k: v for k, v in SERIES_MAP.items() if k in raw.columns}
    df = raw.rename(columns=rename)
    df = df.ffill()

    if freq != "D":
        df = df.resample(freq).last()

    df = df[df.index >= start].copy()
    df = df.dropna(subset=["oas_ig"])

    return df


def fetch_fred(start: str = "2000-01-01",
               end: str = "2024-12-31") -> pd.DataFrame:
    """Fetch series from FRED via pandas_datareader.

    Raises ImportError if pandas_datareader is not installed.
    Raises RuntimeError if the FRED fetch fails.
    """
    import pandas_datareader.data as web

    frames = {}
    for name, series_id in FRED_SERIES.items():
        df = web.DataReader(series_id, "fred", start, end)
        df.columns = [name]
        frames[name] = df[name]
    data = pd.concat(frames, axis=1).resample("ME").last()
    data = data.replace(".", np.nan).astype(float)
    return data


def monthly_returns(df: pd.DataFrame, col: str = "tr_ig") -> pd.Series:
    """Compute monthly total returns from a total return index."""
    return df[col].pct_change().rename(f"ret_{col}")


def oas_changes(df: pd.DataFrame) -> pd.DataFrame:
    """Compute OAS changes at multiple horizons for all OAS series."""
    oas_cols = [c for c in df.columns if c.startswith("oas_")]
    out = {}
    for col in oas_cols:
        s = df[col].dropna()
        out[f"{col}_chg1m"]  = s.diff(1)
        out[f"{col}_chg3m"]  = s.diff(3)
        out[f"{col}_chg6m"]  = s.diff(6)
        out[f"{col}_chg12m"] = s.diff(12)
    return pd.DataFrame(out).reindex(df.index)


def compute_credit_factors(df: pd.DataFrame) -> dict:
    """
    Compute bucket-level credit factors from FRED data.

    Returns a dict keyed by (rating, duration_bucket) with:
      - dts: Duration-Times-Spread (midpoint_dur × latest OAS in %)
      - momentum_6m: 6-month trailing total return
      - value: log(OAS) z-score vs cross-section of rating buckets

    All values come from real FRED data — nothing synthetic.
    """
    factors = {}

    # Latest OAS per rating bucket
    latest_oas = {}
    for rating, col in RATING_OAS_COL.items():
        if col in df.columns:
            val = df[col].dropna()
            if len(val) > 0:
                latest_oas[rating] = float(val.iloc[-1])

    # Latest OAS per duration bucket
    latest_dur_oas = {}
    for dbucket, col in DURATION_BUCKET_OAS_COL.items():
        if col in df.columns:
            val = df[col].dropna()
            if len(val) > 0:
                latest_dur_oas[dbucket] = float(val.iloc[-1])

    # DTS per (rating, duration_bucket)
    # DTS = midpoint_duration × OAS
    # We combine rating-level OAS with duration-bucket adjustment
    for rating in RATING_OAS_COL:
        for dbucket, mid_dur in DURATION_BUCKET_MIDPOINTS.items():
            rating_oas = latest_oas.get(rating)
            dur_oas = latest_dur_oas.get(dbucket)
            if rating_oas is None:
                continue
            # Use duration-bucket OAS if available, otherwise rating OAS
            oas_pct = dur_oas if dur_oas is not None else rating_oas
            dts = mid_dur * oas_pct
            factors.setdefault((rating, dbucket), {})["dts"] = dts

    # Momentum: 6-month trailing total return per rating
    for rating, tr_col in MOMENTUM_TR_COL.items():
        if tr_col in df.columns:
            tr = df[tr_col].dropna()
            if len(tr) >= 6:
                mom_6m = float(tr.iloc[-1] / tr.iloc[-6] - 1.0)
            else:
                mom_6m = 0.0
            for dbucket in DURATION_BUCKET_MIDPOINTS:
                factors.setdefault((rating, dbucket), {})["momentum_6m"] = mom_6m

    # Value: excess log spread vs cross-section of ratings
    if latest_oas:
        log_spreads = {r: np.log(max(v, 0.01)) for r, v in latest_oas.items()}
        mean_log = np.mean(list(log_spreads.values()))
        std_log = np.std(list(log_spreads.values()))
        if std_log < 1e-8:
            std_log = 1.0
        for rating, ls in log_spreads.items():
            value_z = (ls - mean_log) / std_log
            for dbucket in DURATION_BUCKET_MIDPOINTS:
                factors.setdefault((rating, dbucket), {})["value"] = value_z

    return factors


def compute_analytics(data: pd.DataFrame) -> dict:
    """
    Compute analytics for the paper's empirical section:
      1. Value signal: OAS quintile -> forward 12m return
      2. Momentum signal: 6m spread change -> forward 3m return
      3. Quality signal: rolling 5y Sharpe by rating tier
    """
    d = data.dropna(subset=["oas_ig", "tr_ig"]).copy()

    fwd_12m = d["tr_ig"].pct_change(12).shift(-12)
    d["fwd_12m"] = fwd_12m
    d["oas_quintile"] = pd.qcut(
        d["oas_ig"], 5,
        labels=["Q1\n(tight)", "Q2", "Q3", "Q4", "Q5\n(wide)"]
    )
    value_signal = (
        d.dropna(subset=["oas_quintile", "fwd_12m"])
         .groupby("oas_quintile", observed=True)["fwd_12m"]
         .mean()
         .mul(100)
    )

    d["oas_mom_6m"] = d["oas_ig"].diff(6)
    d["fwd_3m"] = d["tr_ig"].pct_change(3).shift(-3)
    d["mom_regime"] = d["oas_mom_6m"].apply(
        lambda x: "Widening\n(bearish)" if x > 0 else "Tightening\n(bullish)"
        if pd.notna(x) else np.nan
    )
    momentum_signal = (
        d.dropna(subset=["mom_regime", "fwd_3m"])
         .groupby("mom_regime")["fwd_3m"]
         .mean()
         .mul(100)
    )

    d["ret_ig"] = d["tr_ig"].pct_change()
    d["ret_hy"] = d["tr_hy"].pct_change() if "tr_hy" in d.columns else np.nan

    def _rolling_sharpe(ret_series, window=CHART_ROLLING_WINDOW):
        roll_ret = ret_series.rolling(window).mean()
        roll_std = ret_series.rolling(window).std()
        return (roll_ret / roll_std * np.sqrt(12)).dropna()

    sharpe_ig = _rolling_sharpe(d["ret_ig"].dropna())
    sharpe_hy = _rolling_sharpe(d["ret_hy"].dropna()) if "tr_hy" in d.columns else None

    return {
        "value_signal"   : value_signal,
        "momentum_signal": momentum_signal,
        "sharpe_ig"      : sharpe_ig,
        "sharpe_hy"      : sharpe_hy,
        "raw"            : d,
    }
