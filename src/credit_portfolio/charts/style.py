"""Shared chart colors and rcParams."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from credit_portfolio.data.constants import COLORS, CHART_DPI, MPL_RCPARAMS

BLUE   = COLORS["primary"]
RED    = COLORS["accent"]
AMBER  = COLORS["amber"]
GREEN  = COLORS["green"]
GRAY   = COLORS["neutral"]

FIG_DPI = CHART_DPI


def apply_style():
    """Apply publication-quality rcParams."""
    plt.rcParams.update(MPL_RCPARAMS)
