"""Attribution, stress testing, Monte Carlo, and LLM commentary generation."""

from credit_portfolio.analytics.attribution import attribute, format_for_llm, AttributionReport
from credit_portfolio.analytics.commentary import generate_commentary, generate_commentary_mock
from credit_portfolio.analytics.stress_test import run_stress_test, StressResult
from credit_portfolio.analytics.monte_carlo import run_monte_carlo, MonteCarloResult
