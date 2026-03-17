"""
Prophet-based BL views: forecasts OAS for each rating bucket and
converts to expected excess returns.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from credit_portfolio.log import get_logger
from credit_portfolio.data.constants import (
    DURATIONS, PROPHET_HORIZON_MONTHS, PROPHET_INTERVAL_WIDTH,
    PROPHET_CHANGEPOINT_PRIOR_SCALE, PROPHET_SEASONALITY_PRIOR_SCALE,
    PROPHET_MCMC_ITER, PROPHET_MIN_MONTHS, OAS_PCT_TO_BP,
    PROPHET_CAP_MULTIPLIER, PROPHET_FLOOR, BUCKET_CONFIG,
)

logger = get_logger(__name__)


def fit_prophet_for_bucket(
    series: pd.Series,
    horizon_months: int = PROPHET_HORIZON_MONTHS,
    changepoint_prior_scale: float = PROPHET_CHANGEPOINT_PRIOR_SCALE,
    seasonality_prior_scale: float = PROPHET_SEASONALITY_PRIOR_SCALE,
    interval_width: float = PROPHET_INTERVAL_WIDTH,
) -> dict:
    """Fit Prophet to a single OAS series and forecast h months ahead."""
    from prophet import Prophet

    df_p = pd.DataFrame({
        "ds": series.index,
        "y" : series.values,
        "cap": series.max() * PROPHET_CAP_MULTIPLIER,
        "floor": PROPHET_FLOOR,
    }).dropna()

    model = Prophet(
        growth="logistic",
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        interval_width=interval_width,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
    )
    model.fit(df_p, iter=PROPHET_MCMC_ITER)

    future = model.make_future_dataframe(periods=horizon_months, freq="ME")
    future["cap"] = df_p["cap"].iloc[0]
    future["floor"] = df_p["floor"].iloc[0]

    forecast = model.predict(future)

    forecast_row = forecast.iloc[-1]
    current_oas  = float(series.dropna().iloc[-1])
    forecast_oas = float(forecast_row["yhat"])
    delta_oas    = forecast_oas - current_oas
    uncertainty  = float((forecast_row["yhat_upper"] - forecast_row["yhat_lower"]) / 2)

    return {
        "current_oas" : current_oas,
        "forecast_oas": forecast_oas,
        "delta_oas"   : delta_oas,
        "uncertainty" : uncertainty,
        "forecast_df" : forecast,
        "model"       : model,
    }


def oas_to_expected_return(current_oas: float, delta_oas: float,
                           duration: float, horizon_months: int = 3) -> float:
    """
    Convert OAS forecast into expected excess return.

    r = carry + price_return
    r = (OAS * horizon/12) + duration * (-delta_oas) / 100
    """
    carry = (current_oas / OAS_PCT_TO_BP) * (horizon_months / 12)
    price_return = duration * (-delta_oas / OAS_PCT_TO_BP) * (horizon_months / 12)
    return carry + price_return


def generate_all_views(
    monthly: pd.DataFrame,
    horizon_months: int = PROPHET_HORIZON_MONTHS,
    changepoint_prior_scale: float = PROPHET_CHANGEPOINT_PRIOR_SCALE,
    seasonality_prior_scale: float = PROPHET_SEASONALITY_PRIOR_SCALE,
    interval_width: float = PROPHET_INTERVAL_WIDTH,
) -> dict:
    """Run Prophet on all five rating buckets and compute expected returns."""
    views = {}
    logger.info("Running Prophet forecasts (this takes ~30 seconds)...")

    for bucket, cfg in BUCKET_CONFIG.items():
        col = cfg["oas_col"]
        if col not in monthly.columns:
            continue

        series = monthly[col].dropna()
        if len(series) < PROPHET_MIN_MONTHS:
            continue

        try:
            result = fit_prophet_for_bucket(
                series, horizon_months,
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale,
                interval_width=interval_width,
            )
            exp_ret = oas_to_expected_return(
                current_oas=result["current_oas"],
                delta_oas=result["delta_oas"],
                duration=cfg["duration"],
                horizon_months=horizon_months,
            )
            views[bucket] = {
                "expected_return": exp_ret,
                "uncertainty"   : result["uncertainty"],
                "current_oas"   : result["current_oas"],
                "forecast_oas"  : result["forecast_oas"],
                "delta_oas"     : result["delta_oas"],
                "delta_oas_bp"  : result["delta_oas"] * OAS_PCT_TO_BP,
            }
            logger.info(
                "%3s: current=%.2f%% forecast=%.2f%% delta=%+.1fbp exp_ret=%+.2f%%",
                bucket, result["current_oas"], result["forecast_oas"],
                result["delta_oas"] * OAS_PCT_TO_BP, exp_ret * OAS_PCT_TO_BP,
            )
        except Exception as e:
            logger.warning("%s: Prophet failed (%s), skipping", bucket, e)

    return views
