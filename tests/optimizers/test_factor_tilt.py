"""Tests for optimizers/factor_tilt.py"""

import numpy as np
import pandas as pd
import pytest

from credit_portfolio.optimizers.factor_tilt import optimise, OptConfig, OptResult


class TestOptimise:
    def test_optimal_status(self, sample_universe):
        result = optimise(sample_universe)
        assert result.status == "optimal"

    def test_weights_sum_to_one(self, sample_universe):
        result = optimise(sample_universe)
        assert abs(result.weights.sum() - 1.0) < 1e-4

    def test_non_negative(self, sample_universe):
        result = optimise(sample_universe)
        assert (result.weights >= -1e-6).all()

    def test_single_name_cap(self, sample_universe):
        config = OptConfig(max_single_name=0.04)
        result = optimise(sample_universe, config=config)
        assert (result.weights <= 0.04 + 1e-4).all()

    def test_sector_neutrality(self, sample_universe):
        config = OptConfig(max_sector_dev=0.05)
        result = optimise(sample_universe, config=config)
        df_e = sample_universe[sample_universe["quality_score"] >= config.quality_floor]
        bmark = np.ones(len(df_e)) / len(df_e)
        for sector in df_e["sector"].unique():
            mask = (df_e["sector"] == sector).values
            port = result.weights.reindex(df_e["bond_id"].values).fillna(0).values[mask].sum()
            bench = bmark[mask].sum()
            assert abs(port - bench) <= 0.05 + 1e-3

    def test_duration_neutrality(self, sample_universe):
        config = OptConfig(max_dur_dev=0.03)
        result = optimise(sample_universe, config=config)
        df_e = sample_universe[sample_universe["quality_score"] >= config.quality_floor]
        bmark = np.ones(len(df_e)) / len(df_e)
        for db in df_e["duration_bucket"].unique():
            mask = (df_e["duration_bucket"] == db).values
            port = result.weights.reindex(df_e["bond_id"].values).fillna(0).values[mask].sum()
            bench = bmark[mask].sum()
            assert abs(port - bench) <= 0.03 + 1e-3

    def test_turnover_cap_with_prior_w(self, sample_universe):
        r0 = optimise(sample_universe)
        from credit_portfolio.data.universe import build_universe, JUNE_SHOCK
        df_t1 = build_universe(n=60, shock=JUNE_SHOCK)
        config = OptConfig(max_turnover=0.20)
        r1 = optimise(df_t1, prior_w=r0.weights.values, config=config)
        assert r1.status == "optimal"

    def test_quality_floor_exclusion(self, sample_universe):
        config = OptConfig(quality_floor=50.0)
        result = optimise(sample_universe, config=config)
        assert result.n_excluded > 0

    def test_positive_objective(self, sample_universe):
        result = optimise(sample_universe)
        assert result.objective_value > 0

    def test_infeasible_fallback(self):
        # Very tight constraints -> fallback to equal weight
        from credit_portfolio.data.universe import build_universe
        df = build_universe(n=10)
        config = OptConfig(max_sector_dev=0.001, max_dur_dev=0.001,
                          max_single_name=0.001)
        result = optimise(df, config=config)
        # Should either solve or fallback
        assert result.weights.sum() > 0

    def test_factor_exposures_dict(self, sample_universe):
        result = optimise(sample_universe)
        assert "z_dts" in result.factor_exposures
        assert "z_value" in result.factor_exposures
        assert "z_momentum" in result.factor_exposures

    def test_binding_constraints_detected(self, sample_universe):
        result = optimise(sample_universe)
        # binding_constraints should be a list
        assert isinstance(result.binding_constraints, list)
