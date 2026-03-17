"""V1 Pipeline: universe -> factor optimizer -> attribution -> commentary."""

import os

from credit_portfolio.config import load_config, resolve_output_dir
from credit_portfolio.log import get_logger
from credit_portfolio.data.constants import DEFAULT_BOND_COUNT, JUNE_SHOCK
from credit_portfolio.data.universe import build_universe
from credit_portfolio.data.loader import fetch_fred, compute_analytics
from credit_portfolio.optimizers.factor_tilt import optimise, OptConfig
from credit_portfolio.analytics.attribution import attribute, format_for_llm
from credit_portfolio.analytics.commentary import generate_commentary, generate_commentary_mock
from credit_portfolio.charts.empirical import (
    chart_value_signal, chart_momentum_signal, chart_quality_sharpe,
)

logger = get_logger(__name__)

BANNER = "=" * 70


def run(output_dir: str | None = None):
    cfg = load_config()
    if output_dir is None:
        output_dir = str(resolve_output_dir(cfg))

    logger.info(BANNER)
    logger.info(" FACTOR CREDIT OPTIMIZER + QUARTERLY COMMENTARY ENGINE")
    logger.info(" Systematic Multi-Factor Investment Grade Credit Strategy")
    logger.info(BANNER)

    logger.info("[1/5] Building IG credit universe (%d bonds)...", DEFAULT_BOND_COUNT)
    df_march = build_universe(n=DEFAULT_BOND_COUNT)
    df_june = build_universe(n=DEFAULT_BOND_COUNT, shock=JUNE_SHOCK)
    logger.info("      March universe: %d bonds", len(df_march))
    logger.info("      June universe : %d bonds", len(df_june))
    logger.info("      Sectors       : %s", ", ".join(sorted(df_march["sector"].unique())))

    logger.info("[2/5] Running portfolio optimisation (CVXPY / CLARABEL)...")
    config = OptConfig()
    result_march = optimise(df_march, config=config)
    result_june = optimise(df_june, prior_w=result_march.weights.values, config=config)
    logger.info("      March status  : %s | excluded: %d", result_march.status, result_march.n_excluded)
    logger.info("      June  status  : %s | excluded: %d", result_june.status, result_june.n_excluded)

    logger.info("[3/5] Computing rebalancing attribution...")
    report = attribute(df_march, df_june, result_march, result_june)
    logger.info(format_for_llm(report))

    logger.info("[4/5] Generating quarterly client commentary...")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        commentary = generate_commentary(report)
    else:
        logger.info("      ANTHROPIC_API_KEY not set — using built-in mock generator.")
        commentary = generate_commentary_mock(report)

    logger.info(BANNER)
    logger.info(" QUARTERLY REBALANCING COMMENTARY")
    logger.info(" %s | Multi-Factor IG Credit Strategy", report.rebal_date)
    logger.info(BANNER)
    logger.info(commentary)
    logger.info(BANNER)

    logger.info("[5/5] Generating FRED empirical charts...")
    data = fetch_fred()
    chart_value_signal(data, output_dir)
    chart_momentum_signal(data, output_dir)
    chart_quality_sharpe(data, output_dir)
    logger.info("      Charts saved to %s/", output_dir)

    logger.info(BANNER)
    logger.info(" PIPELINE COMPLETE")
    logger.info(BANNER)
