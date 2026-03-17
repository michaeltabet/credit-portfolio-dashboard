"""Publication-quality charts for the bucket rotation backtest."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

from credit_portfolio.data.constants import COLORS, CHART_DPI, MPL_RCPARAMS
from credit_portfolio.backtests.bucket_backtest import BacktestResult, RATING_BUCKETS


def _apply_style():
    plt.rcParams.update(MPL_RCPARAMS)


def chart_cumulative_returns(result: BacktestResult, output_dir: str) -> str:
    """Cumulative excess return: strategy vs benchmark."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(result.cumulative_strategy.index, result.cumulative_strategy.values,
            color=COLORS["primary"], linewidth=1.5, label="Factor-Tilted")
    ax.plot(result.cumulative_benchmark.index, result.cumulative_benchmark.values,
            color=COLORS["neutral"], linewidth=1.5, linestyle="--", label="Market-Weight Benchmark")

    ax.set_ylabel("Cumulative Excess Return (growth of $1)")
    ax.set_title("Credit Factor Rotation: Cumulative Performance", fontweight="bold")
    ax.legend(frameon=False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(5))

    # Shade recessions approximately
    _shade_crises(ax)

    fig.tight_layout()
    path = str(Path(output_dir) / "backtest_cumulative.png")
    fig.savefig(path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def chart_rolling_sharpe(result: BacktestResult, output_dir: str,
                         window: int = 36) -> str:
    """Rolling 36-month Sharpe ratio: strategy vs benchmark."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 4))

    for series, label, color, ls in [
        (result.monthly_returns_strategy, "Factor-Tilted", COLORS["primary"], "-"),
        (result.monthly_returns_benchmark, "Benchmark", COLORS["neutral"], "--"),
    ]:
        roll_mean = series.rolling(window).mean() * 12
        roll_std = series.rolling(window).std() * np.sqrt(12)
        sharpe = (roll_mean / roll_std).dropna()
        ax.plot(sharpe.index, sharpe.values, color=color, linewidth=1.2,
                linestyle=ls, label=label)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel(f"Rolling {window}m Sharpe Ratio")
    ax.set_title(f"Rolling {window}-Month Sharpe Ratio", fontweight="bold")
    ax.legend(frameon=False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    path = str(Path(output_dir) / "backtest_rolling_sharpe.png")
    fig.savefig(path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def chart_drawdown(result: BacktestResult, output_dir: str) -> str:
    """Drawdown chart for strategy and benchmark."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 4))

    for cum, label, color in [
        (result.cumulative_strategy, "Factor-Tilted", COLORS["primary"]),
        (result.cumulative_benchmark, "Benchmark", COLORS["neutral"]),
    ]:
        peak = cum.cummax()
        dd = (cum - peak) / peak
        ax.fill_between(dd.index, dd.values, 0, alpha=0.3, color=color, label=label)
        ax.plot(dd.index, dd.values, color=color, linewidth=0.8)

    ax.set_ylabel("Drawdown")
    ax.set_title("Drawdown from Peak", fontweight="bold")
    ax.legend(frameon=False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    import matplotlib.ticker as mtick
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    fig.tight_layout()
    path = str(Path(output_dir) / "backtest_drawdown.png")
    fig.savefig(path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def chart_weight_history(result: BacktestResult, output_dir: str) -> str:
    """Stacked area chart of portfolio weights over time."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 4))

    colors = [COLORS["primary"], COLORS["green"], COLORS["amber"], COLORS["accent"]]
    w = result.weights_history
    ax.stackplot(w.index, *[w[b].values for b in RATING_BUCKETS],
                 labels=RATING_BUCKETS, colors=colors, alpha=0.7)

    ax.set_ylabel("Portfolio Weight")
    ax.set_title("Rating Bucket Allocation Over Time", fontweight="bold")
    ax.legend(loc="upper left", frameon=False, ncol=4)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    path = str(Path(output_dir) / "backtest_weights.png")
    fig.savefig(path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def chart_excess_returns_distribution(result: BacktestResult, output_dir: str) -> str:
    """Histogram of monthly excess returns (strategy - benchmark)."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 4))

    excess = (result.monthly_returns_strategy - result.monthly_returns_benchmark) * 100
    ax.hist(excess.values, bins=40, color=COLORS["primary"], alpha=0.7, edgecolor="white")
    ax.axvline(excess.mean(), color=COLORS["accent"], linewidth=1.5, linestyle="--",
               label=f"Mean: {excess.mean():+.1f}bp/month")
    ax.axvline(0, color="black", linewidth=0.5)

    ax.set_xlabel("Monthly Excess Return vs Benchmark (bp)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Monthly Alpha", fontweight="bold")
    ax.legend(frameon=False)

    fig.tight_layout()
    path = str(Path(output_dir) / "backtest_alpha_dist.png")
    fig.savefig(path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def chart_all(result: BacktestResult, output_dir: str) -> list[str]:
    """Generate all backtest charts. Returns list of file paths."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    paths = [
        chart_cumulative_returns(result, output_dir),
        chart_rolling_sharpe(result, output_dir),
        chart_drawdown(result, output_dir),
        chart_weight_history(result, output_dir),
        chart_excess_returns_distribution(result, output_dir),
    ]
    return paths


def _shade_crises(ax):
    """Shade approximate crisis periods for context."""
    crises = [
        ("2007-06", "2009-06"),  # GFC
        ("2020-02", "2020-06"),  # COVID
    ]
    for start, end in crises:
        try:
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                       alpha=0.08, color="red", zorder=0)
        except Exception:
            pass
