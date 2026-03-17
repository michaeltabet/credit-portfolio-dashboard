"""Chart generation for empirical signals, regimes, and BL results."""

from credit_portfolio.charts.empirical import (
    chart_value_signal, chart_momentum_signal, chart_quality_sharpe,
)
from credit_portfolio.charts.regime import chart_hmm_regimes
from credit_portfolio.charts.bl_charts import chart_bl_posterior, chart_architecture
