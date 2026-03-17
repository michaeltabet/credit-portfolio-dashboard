"""Figure 5: BL posterior vs prior + architecture diagram."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from credit_portfolio.charts.style import BLUE, RED, AMBER, GREEN, GRAY, FIG_DPI, apply_style
from credit_portfolio.data.constants import CHART_FILENAMES


def chart_bl_posterior(assets, mu_prior, mu_posterior, Q_vec,
                       regime_label, omega_scale,
                       output_dir: str = "output") -> str:
    """Figure 5a: BL posterior vs prior and views bar chart."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(assets))
    width = 0.28
    labels = [a.replace("oas_", "").upper() if "oas_" in a else a for a in assets]

    ax.bar(x - width, mu_prior * 100, width, label="Prior (equilibrium)",
           color="#B5C7D9", edgecolor="white")
    ax.bar(x, Q_vec * 100, width, label="View (trend signal)",
           color="#7A9DBF", edgecolor="white")
    ax.bar(x + width, mu_posterior * 100, width, label="BL posterior",
           color=BLUE, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Expected excess return (%)")
    ax.set_title(f"Figure 5: Black-Litterman posterior vs prior and views\n"
                 f"Regime: {regime_label}  |  Omega scale: {omega_scale}x",
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.3, ls="--")
    fig.tight_layout()
    p = f"{output_dir}/{CHART_FILENAMES['bl_posterior']}"
    fig.savefig(p, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return p


def chart_architecture(output_dir: str = "output") -> str:
    """Figure 5b: Three-layer pipeline architecture diagram."""
    apply_style()
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 3.5)
    ax.axis("off")

    boxes = [
        (0.5, 1.5, 1.8, 1.4, BLUE,    "Layer 1\nHMM Regime",
         "3 hidden states\nCompress/Normal/Stress\nOutputs tau"),
        (3.2, 1.5, 1.8, 1.4, AMBER,   "Layer 2\nProphet + BL",
         "OAS forecasts per\nrating bucket\nBL posterior mu_BL"),
        (5.9, 1.5, 1.8, 1.4, GREEN,   "Layer 3\nCVXPY Optimizer",
         "Max w'mu_BL - risk\nSector/duration\nneutrality"),
        (0.5, 0.1, 1.8, 0.7, "#AED6F1", "Real FRED data",
         "349 months, 1997-2026"),
        (3.2, 0.1, 1.8, 0.7, "#FAD7A0", "tau modulates Omega",
         "Stress=high uncertainty"),
        (5.9, 0.1, 1.8, 0.7, "#ABEBC6", "BL weights",
         "Replace Z-score objective"),
    ]

    for x_pos, y, w, h, color, title, sub in boxes:
        rect = plt.Rectangle((x_pos, y), w, h, facecolor=color, edgecolor="white",
                              linewidth=1.5, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x_pos + w / 2, y + h * 0.65, title, ha="center", va="center",
                fontsize=9, fontweight="bold",
                color="white" if color in [BLUE, AMBER, GREEN] else "#333")
        ax.text(x_pos + w / 2, y + h * 0.25, sub, ha="center", va="center",
                fontsize=7.5,
                color="white" if color in [BLUE, AMBER, GREEN] else "#555")

    for x1, x2 in [(2.3, 3.2), (5.0, 5.9)]:
        ax.annotate("", xy=(x2, 2.2), xytext=(x1, 2.2),
                    arrowprops=dict(arrowstyle="->", color="#333", lw=1.5))

    ax.text(2.75, 2.35, "tau", ha="center", fontsize=9, color="#333")
    ax.text(5.45, 2.35, "mu_BL\nSigma_BL", ha="center", fontsize=8, color="#333")

    ax.set_title("Figure 5: Three-layer portfolio construction architecture\n"
                 "HMM -> Prophet/Black-Litterman -> CVXPY Optimizer",
                 fontsize=10, loc="left", pad=8)

    fig.tight_layout()
    p = f"{output_dir}/{CHART_FILENAMES['architecture']}"
    fig.savefig(p, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return p
