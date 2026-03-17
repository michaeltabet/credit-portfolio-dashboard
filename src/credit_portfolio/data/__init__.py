"""Data loading, universe generation, and shared constants."""

from credit_portfolio.data.loader import load, fetch_fred, compute_analytics
from credit_portfolio.data.universe import build_universe
from credit_portfolio.data.constants import (
    SERIES_MAP, DURATIONS, MARKET_WEIGHTS, IG_MARKET_WEIGHTS,
    REGIME_LABELS, OMEGA_SCALE, SECTORS, RATINGS, JUNE_SHOCK,
)
