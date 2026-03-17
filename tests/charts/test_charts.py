"""Tests for charts modules."""

import os
import pytest

from credit_portfolio.charts.empirical import (
    chart_value_signal, chart_momentum_signal, chart_quality_sharpe,
)
from credit_portfolio.charts.regime import chart_hmm_regimes
from credit_portfolio.charts.bl_charts import chart_bl_posterior, chart_architecture


class TestEmpiricalCharts:
    def test_creates_three_pngs(self, fred_csv_path, tmp_output_dir):
        from credit_portfolio.data.loader import load
        df = load(fred_csv_path)
        p1 = chart_value_signal(df, tmp_output_dir)
        p2 = chart_momentum_signal(df, tmp_output_dir)
        p3 = chart_quality_sharpe(df, tmp_output_dir)
        assert os.path.exists(p1)
        assert os.path.exists(p2)
        assert os.path.exists(p3)


class TestRegimeChart:
    def test_creates_png(self, fred_csv_path, tmp_output_dir):
        from credit_portfolio.data.loader import load
        from credit_portfolio.models.hmm_regime import fit_hmm
        df = load(fred_csv_path)
        hmm_result = fit_hmm(df)
        p = chart_hmm_regimes(df, hmm_result, tmp_output_dir)
        assert os.path.exists(p)


class TestBLCharts:
    def test_bl_chart_creates_png(self, tmp_output_dir):
        import numpy as np
        assets = ["AAA", "AA", "A", "BBB"]
        pi = np.array([0.01, 0.02, 0.03, 0.04])
        mu = np.array([0.015, 0.025, 0.035, 0.045])
        Q = np.array([0.02, 0.03, 0.04, 0.05])
        p = chart_bl_posterior(assets, pi, mu, Q, "NORMAL", 1.0, tmp_output_dir)
        assert os.path.exists(p)

    def test_architecture_creates_png(self, tmp_output_dir):
        p = chart_architecture(tmp_output_dir)
        assert os.path.exists(p)

    def test_all_figures_closed(self, tmp_output_dir):
        import matplotlib.pyplot as plt
        import numpy as np
        assets = ["AAA", "AA", "A", "BBB"]
        pi = np.array([0.01, 0.02, 0.03, 0.04])
        mu = pi * 1.1
        Q = pi * 1.2
        chart_bl_posterior(assets, pi, mu, Q, "NORMAL", 1.0, tmp_output_dir)
        chart_architecture(tmp_output_dir)
        assert len(plt.get_fignums()) == 0
