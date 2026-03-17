"""Tests for optimizers/mean_variance_bl.py"""

import numpy as np
import pandas as pd
import pytest

from credit_portfolio.optimizers.mean_variance_bl import optimise_bl, map_bl_returns_to_bonds


class TestOptimiseBL:
    def test_map_returns_shape(self, sample_universe, sample_bl_result):
        mu = map_bl_returns_to_bonds(sample_universe, sample_bl_result)
        assert len(mu) == len(sample_universe)

    def test_factor_alpha_included(self, sample_universe, sample_bl_result):
        mu = map_bl_returns_to_bonds(sample_universe, sample_bl_result)
        # With non-zero z_composite, returns should vary
        assert np.std(mu) > 0

    def test_optimal_status(self, sample_universe, sample_bl_result):
        result = optimise_bl(sample_universe, sample_bl_result)
        assert result.status == "optimal"

    def test_weights_sum_to_one(self, sample_universe, sample_bl_result):
        result = optimise_bl(sample_universe, sample_bl_result)
        assert abs(result.weights.sum() - 1.0) < 1e-4

    def test_sector_neutrality(self, sample_universe, sample_bl_result):
        result = optimise_bl(sample_universe, sample_bl_result, max_sector_dev=0.05)
        df_e = sample_universe[sample_universe["quality_score"] >= 35.0]
        n = len(df_e)
        bmark = np.ones(n) / n
        for sector in df_e["sector"].unique():
            mask = (df_e["sector"] == sector).values
            port = result.weights.reindex(df_e["bond_id"].values).fillna(0).values[mask].sum()
            bench = bmark[mask].sum()
            assert abs(port - bench) <= 0.05 + 1e-3

    def test_expected_returns_series_length(self, sample_universe, sample_bl_result):
        result = optimise_bl(sample_universe, sample_bl_result)
        assert len(result.expected_returns) == len(result.weights)

    def test_regime_label_propagated(self, sample_universe, sample_bl_result):
        result = optimise_bl(sample_universe, sample_bl_result)
        assert result.regime == sample_bl_result.regime
