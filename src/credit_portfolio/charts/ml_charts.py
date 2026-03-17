"""
ML factor model visualizations: SHAP analysis, feature importance,
walk-forward performance, and time-varying factor weights.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

from credit_portfolio.charts.style import (
    BLUE, RED, AMBER, GREEN, GRAY, FIG_DPI, apply_style,
)
from credit_portfolio.data.constants import CHART_FILENAMES


def chart_shap_summary(
    shap_values: np.ndarray,
    feature_names: list,
    df_current: pd.DataFrame,
    output_dir: str = "output",
) -> str:
    """SHAP beeswarm/bar plot for current universe."""
    import shap

    apply_style()
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(
        shap_values,
        df_current[feature_names] if all(f in df_current.columns for f in feature_names)
        else pd.DataFrame(np.zeros((shap_values.shape[0], len(feature_names))),
                          columns=feature_names),
        feature_names=[f.replace("z_", "").title() for f in feature_names],
        show=False,
        plot_size=None,
    )
    plt.title(
        "ML Factor Model: SHAP Feature Importance\n"
        "Each dot = one bond; color = feature value; x = SHAP impact",
        fontsize=10, loc="left",
    )
    plt.tight_layout()
    p = f"{output_dir}/{CHART_FILENAMES['shap_summary']}"
    plt.savefig(p, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    return p


def chart_shap_weights_over_time(
    shap_history: pd.DataFrame,
    output_dir: str = "output",
) -> str:
    """Stacked area chart of SHAP-derived factor weights through time."""
    apply_style()
    os.makedirs(output_dir, exist_ok=True)

    if shap_history.empty:
        return ""

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = [BLUE, RED, AMBER, GREEN, GRAY]
    cols = shap_history.columns.tolist()

    ax.stackplot(
        shap_history.index,
        *[shap_history[c].values for c in cols],
        labels=[c.replace("z_", "").title() for c in cols],
        colors=colors[:len(cols)],
        alpha=0.8,
    )

    ax.set_ylabel("Factor Weight (from SHAP)")
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.set_title(
        "Time-Varying Factor Weights (SHAP-derived)\n"
        "Replaces fixed weights -- captures nonlinear regime dependence",
        fontsize=10, loc="left",
    )
    fig.autofmt_xdate()
    fig.tight_layout()
    p = f"{output_dir}/{CHART_FILENAMES['shap_weights_time']}"
    fig.savefig(p, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return p


def chart_walk_forward_performance(
    folds: list,
    output_dir: str = "output",
) -> str:
    """Walk-forward OOS R-squared and rank IC over time."""
    apply_style()
    os.makedirs(output_dir, exist_ok=True)

    if not folds:
        return ""

    dates = [pd.Timestamp(f.test_period) for f in folds]
    r2s = [f.r2_oos for f in folds]
    ics = [f.ic_rank for f in folds]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.bar(dates, r2s, width=25, color=BLUE, alpha=0.7)
    ax1.axhline(0, color="black", lw=0.5)
    ax1.set_ylabel("OOS R-squared")
    ax1.set_title(
        "Walk-Forward Cross-Validation Performance\n"
        "Expanding window with purge gap",
        fontsize=10, loc="left",
    )

    ax2.bar(dates, ics, width=25, color=GREEN, alpha=0.7)
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_ylabel("Rank IC (Spearman)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    fig.autofmt_xdate()
    fig.tight_layout()
    p = f"{output_dir}/{CHART_FILENAMES['walkforward_perf']}"
    fig.savefig(p, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return p


def chart_factor_weights_comparison(
    fixed_weights: dict,
    shap_weights: dict,
    output_dir: str = "output",
) -> str:
    """Side-by-side comparison: fixed weights vs SHAP-derived weights."""
    apply_style()
    os.makedirs(output_dir, exist_ok=True)

    factors = sorted(set(fixed_weights.keys()) | set(shap_weights.keys()))
    labels = [f.replace("z_", "").title() for f in factors]

    fixed_vals = [fixed_weights.get(f, 0) for f in factors]
    shap_vals = [shap_weights.get(f, 0) for f in factors]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(factors))
    width = 0.35

    ax.bar(x - width / 2, fixed_vals, width, label="Fixed", color=GRAY)
    ax.bar(x + width / 2, shap_vals, width, label="SHAP-derived", color=BLUE)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Factor Weight")
    ax.set_title(
        "Factor Weights: Fixed vs ML-Derived (SHAP)\n"
        "ML captures regime-dependent factor importance",
        fontsize=10, loc="left",
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    p = f"{output_dir}/{CHART_FILENAMES['weight_comparison']}"
    fig.savefig(p, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return p


def chart_ml_vs_bl_returns(
    ml_predictions: pd.Series,
    bl_returns: pd.Series,
    output_dir: str = "output",
    top_n: int = 20,
) -> str:
    """Scatter: ML predicted returns vs BL returns per bond."""
    apply_style()
    os.makedirs(output_dir, exist_ok=True)

    common = ml_predictions.index.intersection(bl_returns.index)[:top_n]
    if len(common) == 0:
        return ""

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        bl_returns.loc[common] * 100,
        ml_predictions.loc[common] * 100,
        c=BLUE, alpha=0.6, s=40,
    )

    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "--", color=GRAY, lw=0.8)

    ax.set_xlabel("BL Expected Return (%)")
    ax.set_ylabel("ML Predicted Return (%)")
    ax.set_title(
        "ML vs Black-Litterman Expected Returns\n"
        "Diagonal = perfect agreement",
        fontsize=10, loc="left",
    )
    fig.tight_layout()
    p = f"{output_dir}/{CHART_FILENAMES['ml_vs_bl_returns']}"
    fig.savefig(p, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return p
