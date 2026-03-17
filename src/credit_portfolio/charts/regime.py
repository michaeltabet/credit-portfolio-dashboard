"""Figure 4: HMM regime classification chart."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from credit_portfolio.charts.style import FIG_DPI, apply_style
from credit_portfolio.data.constants import (
    REGIME_LABELS, REGIME_COLORS, HISTORICAL_EVENTS, HMM_BAR_WIDTH,
    OAS_PCT_TO_BP, CHART_FILENAMES,
)


def chart_hmm_regimes(df: pd.DataFrame, hmm_result,
                      output_dir: str = "output") -> str:
    """Two-panel chart: OAS with regime shading + state probabilities."""
    apply_style()

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 6),
        gridspec_kw={"height_ratios": [2.2, 1]}, sharex=True
    )

    oas = df["oas_ig"].dropna() * OAS_PCT_TO_BP  # bp
    states = hmm_result.states.reindex(oas.index).ffill()

    def find_blocks(mask):
        blocks, in_b, s = [], False, None
        items = list(mask.items())
        for i, (ix, v) in enumerate(items):
            if v and not in_b:
                in_b, s = True, ix
            elif not v and in_b:
                in_b = False
                blocks.append((s, items[i - 1][0]))
        if in_b:
            blocks.append((s, items[-1][0]))
        return blocks

    for rid, color in REGIME_COLORS.items():
        mask = states == rid
        blocks = find_blocks(mask)
        used = False
        for s_dt, e_dt in blocks:
            ax1.axvspan(s_dt, e_dt, alpha=0.15, color=color,
                        label=REGIME_LABELS[rid] if not used else None, zorder=1)
            used = True

    ax1.plot(oas.index, oas.values, color="#1A1A1A", lw=0.9, zorder=3)

    for ds, lbl in HISTORICAL_EVENTS:
        dt = pd.Timestamp(ds)
        if oas.index[0] <= dt <= oas.index[-1]:
            ax1.axvline(dt, color="#999", lw=0.6, ls=":", alpha=0.9, zorder=2)
            ax1.text(dt, oas.max() * 0.87, lbl, fontsize=7,
                     color="#555", rotation=90, va="top", zorder=4)

    ax1.set_ylabel("IG OAS (basis points)")
    ax1.legend(loc="upper left", frameon=False, fontsize=9)
    ax1.set_title(
        "Figure 4: Credit market regimes — 3-state Hidden Markov Model\n"
        "ICE BofA US Corporate IG OAS  |  Real FRED data",
        fontsize=10, loc="left"
    )
    ax1.grid(axis="y", alpha=0.2, ls="--")

    probs = hmm_result.state_probs.reindex(oas.index).ffill().fillna(0)
    bottom = np.zeros(len(probs))
    for rid in [0, 1, 2]:
        if rid in probs.columns:
            v = probs[rid].values
            ax2.bar(probs.index, v, bottom=bottom, width=HMM_BAR_WIDTH,
                    color=REGIME_COLORS[rid], alpha=0.75, label=REGIME_LABELS[rid])
            bottom += v

    ax2.set_ylim(0, 1)
    ax2.set_ylabel("State probability")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax2.grid(axis="y", alpha=0.2, ls="--")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator(4))

    fig.tight_layout()
    p = f"{output_dir}/{CHART_FILENAMES['hmm_regimes']}"
    fig.savefig(p, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return p
