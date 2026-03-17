"""Single CLI entry point for the credit portfolio system."""

import argparse
import sys

from credit_portfolio.config import load_config, resolve_output_dir
from credit_portfolio.log import get_logger
from credit_portfolio.data.constants import ML_MODEL_CHOICES

logger = get_logger(__name__)


def main(args=None):
    cfg = load_config()
    default_output = str(resolve_output_dir(cfg))

    parser = argparse.ArgumentParser(
        prog="credit-portfolio",
        description="Factor Credit Portfolio Construction System",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Factor pipeline
    factor_parser = subparsers.add_parser(
        "factor", help="Run V1 factor tilt pipeline"
    )
    factor_parser.add_argument("--output", default=None, help="Output directory")

    # BL pipeline
    bl_parser = subparsers.add_parser(
        "bl", help="Run V2 Black-Litterman pipeline"
    )
    bl_parser.add_argument("--output", default=None, help="Output directory")

    # ML pipeline
    ml_parser = subparsers.add_parser(
        "ml", help="Run V3 ML-enhanced factor pipeline"
    )
    ml_parser.add_argument("--output", default=None, help="Output directory")
    ml_parser.add_argument(
        "--model", default=None,
        choices=ML_MODEL_CHOICES,
        help="ML model type",
    )

    # Backtest
    bt_parser = subparsers.add_parser(
        "backtest", help="Run rolling walk-forward ML backtest"
    )
    bt_parser.add_argument("--output", default=None, help="Output directory")
    bt_parser.add_argument(
        "--model", default=None,
        choices=ML_MODEL_CHOICES,
        help="ML model type",
    )
    bt_parser.add_argument("--periods", type=int, default=None, help="Number of months")

    # Charts only
    charts_parser = subparsers.add_parser(
        "charts", help="Generate empirical charts only"
    )
    charts_parser.add_argument("--output", default=None, help="Output directory")

    parsed = parser.parse_args(args)

    if parsed.command == "factor":
        from credit_portfolio.pipelines.factor_pipeline import run
        run(output_dir=parsed.output)

    elif parsed.command == "bl":
        from credit_portfolio.pipelines.bl_pipeline import run
        run(output_dir=parsed.output)

    elif parsed.command == "ml":
        from credit_portfolio.pipelines.ml_pipeline import run
        run(output_dir=parsed.output, model_type=parsed.model)

    elif parsed.command == "backtest":
        from credit_portfolio.backtests.ml_backtest import (
            run_ml_backtest, MLBacktestConfig, format_stats_table,
        )
        from credit_portfolio.charts.ml_backtest_charts import chart_all
        bt_config = MLBacktestConfig()
        if parsed.model:
            bt_config.model_type = parsed.model
        if parsed.periods:
            bt_config.n_periods = parsed.periods
        result = run_ml_backtest(bt_config)
        logger.info(format_stats_table(result))
        output = parsed.output or default_output
        chart_all(result, output)
        logger.info("Charts saved to %s/", output)

    elif parsed.command == "charts":
        from credit_portfolio.data.loader import fetch_fred
        from credit_portfolio.charts.empirical import (
            chart_value_signal, chart_momentum_signal, chart_quality_sharpe,
        )
        output = parsed.output or default_output
        data = fetch_fred()
        chart_value_signal(data, output)
        chart_momentum_signal(data, output)
        chart_quality_sharpe(data, output)
        logger.info("Charts saved to %s/", output)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
