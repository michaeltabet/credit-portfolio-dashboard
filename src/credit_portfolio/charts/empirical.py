"""Figures 1-3: Value, momentum, and quality signals from FRED data."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from credit_portfolio.charts.style import BLUE, RED, GRAY, FIG_DPI, apply_style
from credit_portfolio.data.constants import CHART_ROLLING_WINDOW, CHART_FILENAMES


def chart_value_signal(data: pd.DataFrame, output_dir: str = "output") -> str:
    """Figure 1: OAS quintile -> forward 12m return."""
    apply_style()
    m = data.dropna(subset=["oas_ig", "tr_ig"]).copy()
    m["fwd_12m"] = m["tr_ig"].pct_change(12).shift(-12)
    m = m.dropna(subset=["fwd_12m"])
    m["quintile"] = pd.qcut(
        m["oas_ig"], 5,
        labels=["Q1\n(tight)", "Q2", "Q3", "Q4", "Q5\n(wide)"]
    )
    vs = m.groupby("quintile", observed=True)["fwd_12m"].mean().mul(100)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [GRAY] * 4 + [BLUE]
    bars = ax.bar(vs.index, vs.values, color=colors, width=0.6, edgecolor="white")
    ax.axhline(0, color="black", lw=0.5)
    for bar, val in zip(bars, vs.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("OAS quintile (Q1=tightest, Q5=widest)")
    ax.set_ylabel("Avg 12m forward IG total return (%)")
    ax.set_title("Figure 1: Value signal — ICE BofA real data\n"
                 "Average 12m forward return by OAS quintile, 1997–2024",
                 fontsize=10, loc="left")
    fig.tight_layout()
    p = f"{output_dir}/{CHART_FILENAMES['value_signal']}"
    fig.savefig(p, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return p


def chart_momentum_signal(data: pd.DataFrame, output_dir: str = "output") -> str:
    """Figure 2: 6m spread change regime -> forward 3m return."""
    apply_style()
    m = data.dropna(subset=["oas_ig", "tr_ig"]).copy()
    m["fwd_3m"] = m["tr_ig"].pct_change(3).shift(-3)
    m["oas_mom"] = m["oas_ig"].diff(6)
    m = m.dropna(subset=["fwd_3m", "oas_mom"])
    m["regime_mom"] = m["oas_mom"].apply(
        lambda x: "Widening\n(bearish)" if x > 0 else "Tightening\n(bullish)")
    ms = m.groupby("regime_mom")["fwd_3m"].mean().mul(100)
    ms = ms.reindex(["Widening\n(bearish)", "Tightening\n(bullish)"])

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(ms.index, ms.values, color=[RED, BLUE], width=0.5, edgecolor="white")
    ax.axhline(0, color="black", lw=0.5)
    for bar, val in zip(bars, ms.values):
        ypos = bar.get_height() + 0.05 if val >= 0 else bar.get_height() - 0.25
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Avg 3m forward IG total return (%)")
    ax.set_title("Figure 2: Momentum signal — real ICE BofA data\n"
                 "Avg 3m forward return by 6m spread-change regime",
                 fontsize=10, loc="left")
    fig.tight_layout()
    p = f"{output_dir}/{CHART_FILENAMES['momentum_signal']}"
    fig.savefig(p, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return p


def chart_quality_sharpe(data: pd.DataFrame, output_dir: str = "output") -> str:
    """Figure 3: Rolling 5y Sharpe ratio, IG vs HY."""
    apply_style()
    fig, ax = plt.subplots(figsize=(9, 4))

    for col, label, color, ls in [
        ("tr_ig", "IG Corporate", BLUE, "-"),
        ("tr_hy", "High Yield", RED, "--"),
    ]:
        if col not in data.columns:
            continue
        r = data[col].pct_change().dropna()
        roll = (r.rolling(CHART_ROLLING_WINDOW).mean()
                / r.rolling(CHART_ROLLING_WINDOW).std()
                * np.sqrt(12)).dropna()
        ax.plot(roll.index, roll.values, color=color, lw=1.2, ls=ls, label=label)

    ax.axhline(0, color="black", lw=0.5)
    for yr, lbl in [(2008, "GFC"), (2020, "COVID")]:
        ax.axvline(pd.Timestamp(f"{yr}-01-01"), color=GRAY, lw=0.8, ls=":", alpha=0.7)
        ax.text(pd.Timestamp(f"{yr}-04-01"), ax.get_ylim()[0] + 0.05,
                lbl, fontsize=8, color=GRAY)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    ax.set_ylabel("Rolling 5-year annualised Sharpe ratio")
    ax.set_title("Figure 3: Quality signal — rolling Sharpe, IG vs HY (real data)\n"
                 "ICE BofA total return indices, 2002–2026",
                 fontsize=10, loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    p = f"{output_dir}/{CHART_FILENAMES['quality_sharpe']}"
    fig.savefig(p, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return p
