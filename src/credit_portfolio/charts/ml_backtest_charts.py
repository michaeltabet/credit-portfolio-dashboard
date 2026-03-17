"""Charts for the ML walk-forward backtest: 3-strategy comparison."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
from pathlib import Path

from credit_portfolio.data.constants import COLORS, CHART_DPI, MPL_RCPARAMS


def _apply_style():
    plt.rcParams.update(MPL_RCPARAMS)


BLUE = COLORS["primary"]
RED = COLORS["accent"]
GREEN = COLORS["green"]
AMBER = COLORS["amber"]
GRAY = COLORS["neutral"]


def chart_cumulative_three_way(result, output_dir: str) -> str:
    """ML vs Fixed vs Benchmark cumulative returns."""
    _apply_style()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(result.cumulative_ml.index, result.cumulative_ml.values,
            color=BLUE, lw=1.8, label="ML-Weighted")
    ax.plot(result.cumulative_fixed.index, result.cumulative_fixed.values,
            color=AMBER, lw=1.5, linestyle="--", label="Fixed-Weight")
    ax.plot(result.cumulative_benchmark.index, result.cumulative_benchmark.values,
            color=GRAY, lw=1.2, linestyle=":", label="Equal-Weight Benchmark")

    ax.set_ylabel("Cumulative Return (growth of $1)")
    ax.set_title("ML Backtest: Cumulative Performance\n"
                 "ML-weighted vs Fixed-weight vs Equal-weight",
                 fontsize=10, loc="left")
    ax.legend(frameon=False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    p = str(Path(output_dir) / "bt_cumulative.png")
    fig.savefig(p, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    return p


def chart_rolling_sharpe_three_way(result, output_dir: str, window: int = 12) -> str:
    """Rolling Sharpe for all 3 strategies."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 4))

    for series, label, color, ls in [
        (result.monthly_returns_ml, "ML-Weighted", BLUE, "-"),
        (result.monthly_returns_fixed, "Fixed-Weight", AMBER, "--"),
        (result.monthly_returns_benchmark, "Benchmark", GRAY, ":"),
    ]:
        roll_mean = series.rolling(window).mean() * 12
        roll_std = series.rolling(window).std() * np.sqrt(12)
        sharpe = (roll_mean / roll_std).dropna()
        if len(sharpe) > 0:
            ax.plot(sharpe.index, sharpe.values, color=color, lw=1.2,
                    linestyle=ls, label=label)

    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel(f"Rolling {window}m Sharpe")
    ax.set_title(f"Rolling {window}-Month Sharpe Ratio", fontsize=10, loc="left")
    ax.legend(frameon=False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    p = str(Path(output_dir) / "bt_rolling_sharpe.png")
    fig.savefig(p, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    return p


def chart_drawdown_comparison(result, output_dir: str) -> str:
    """Drawdown for ML vs Fixed strategies."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 4))

    for cum, label, color in [
        (result.cumulative_ml, "ML-Weighted", BLUE),
        (result.cumulative_fixed, "Fixed-Weight", AMBER),
    ]:
        peak = cum.cummax()
        dd = (cum - peak) / peak
        ax.fill_between(dd.index, dd.values, 0, alpha=0.3, color=color, label=label)
        ax.plot(dd.index, dd.values, color=color, lw=0.8)

    ax.set_ylabel("Drawdown")
    ax.set_title("Drawdown from Peak", fontsize=10, loc="left")
    ax.legend(frameon=False)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    p = str(Path(output_dir) / "bt_drawdown.png")
    fig.savefig(p, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    return p


def chart_factor_attribution_over_time(result, output_dir: str) -> str:
    """Stacked bar: factor return contribution per quarter."""
    _apply_style()
    attrib = result.factor_attribution.copy()

    # Drop 'total' column for stacking
    factor_cols = [c for c in attrib.columns if c != "total"]
    if not factor_cols:
        return ""

    # Resample to quarterly for readability
    q = attrib[factor_cols].resample("QE").sum()

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [BLUE, GREEN, AMBER, RED, GRAY]
    bottom = np.zeros(len(q))

    for i, col in enumerate(factor_cols):
        vals = q[col].values
        color = colors[i % len(colors)]
        ax.bar(q.index, vals, bottom=bottom, width=60,
               label=col.replace("z_", "").title(), color=color, alpha=0.8)
        bottom += vals

    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Factor Return Contribution")
    ax.set_title("Quarterly Factor Attribution (ML Strategy)",
                 fontsize=10, loc="left")
    ax.legend(frameon=False, ncol=len(factor_cols))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    p = str(Path(output_dir) / "bt_factor_attribution.png")
    fig.savefig(p, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    return p


def chart_shap_weights_vs_turnover(result, output_dir: str) -> str:
    """Dual-axis: SHAP weights (stacked area) + turnover (line)."""
    _apply_style()
    shap_df = result.shap_weights_history
    if shap_df.empty:
        return ""

    fig, ax1 = plt.subplots(figsize=(10, 5))

    cols = shap_df.columns.tolist()
    colors = [BLUE, GREEN, AMBER, RED, GRAY]
    ax1.stackplot(
        shap_df.index,
        *[shap_df[c].values for c in cols],
        labels=[c.replace("z_", "").title() for c in cols],
        colors=colors[:len(cols)],
        alpha=0.7,
    )
    ax1.set_ylabel("SHAP Factor Weight")
    ax1.set_ylim(0, 1)
    ax1.legend(loc="upper left", frameon=False, fontsize=8)

    # Overlay turnover on right axis
    ax2 = ax1.twinx()
    ax2.plot(result.turnover_ml.index, result.turnover_ml.values,
             color="black", lw=1.0, alpha=0.5, label="Turnover (ML)")
    ax2.set_ylabel("Monthly Turnover")
    ax2.set_ylim(0, 0.3)

    ax1.set_title("SHAP Factor Weights & Portfolio Turnover Over Time",
                  fontsize=10, loc="left")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    p = str(Path(output_dir) / "bt_shap_turnover.png")
    fig.savefig(p, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    return p


def chart_alpha_distribution(result, output_dir: str) -> str:
    """Histogram of monthly alpha (ML - benchmark)."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 4))

    excess = (result.monthly_returns_ml - result.monthly_returns_benchmark) * 10000  # bp
    ax.hist(excess.values, bins=30, color=BLUE, alpha=0.7, edgecolor="white")
    ax.axvline(excess.mean(), color=RED, lw=1.5, linestyle="--",
               label=f"Mean: {excess.mean():+.1f}bp/month")
    ax.axvline(0, color="black", lw=0.5)

    ax.set_xlabel("Monthly Alpha vs Benchmark (bp)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Monthly Alpha (ML Strategy)",
                 fontsize=10, loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    p = str(Path(output_dir) / "bt_alpha_dist.png")
    fig.savefig(p, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    return p


def chart_all(result, output_dir: str) -> list[str]:
    """Generate all backtest charts."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return [
        chart_cumulative_three_way(result, output_dir),
        chart_rolling_sharpe_three_way(result, output_dir),
        chart_drawdown_comparison(result, output_dir),
        chart_factor_attribution_over_time(result, output_dir),
        chart_shap_weights_vs_turnover(result, output_dir),
        chart_alpha_distribution(result, output_dir),
    ]
