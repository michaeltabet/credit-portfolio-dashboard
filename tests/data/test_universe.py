"""Tests for data/universe.py"""

import numpy as np
import pandas as pd
import pytest

from credit_portfolio.data.universe import build_universe
from credit_portfolio.data.constants import JUNE_SHOCK, OPT_FACTOR_WEIGHTS


FACTOR_COLS = list(OPT_FACTOR_WEIGHTS.keys())  # z_dts, z_value, z_momentum


class TestBuildUniverse:
    def test_shape(self, sample_universe):
        assert sample_universe.shape[0] == 60
        expected_cols = {"bond_id", "sector", "rating", "duration_bucket",
                         "oas_bp", "z_composite"}
        expected_cols.update(FACTOR_COLS)
        assert expected_cols.issubset(set(sample_universe.columns))

    def test_zscore_mean_approx_zero(self, sample_universe):
        for col in FACTOR_COLS:
            for bucket in sample_universe["duration_bucket"].unique():
                mask = sample_universe["duration_bucket"] == bucket
                vals = sample_universe.loc[mask, col]
                if len(vals) > 2:
                    assert abs(vals.mean()) < 0.3, f"{col} mean not ~0 in {bucket}"

    def test_zscore_std_approx_one(self, sample_universe):
        for col in FACTOR_COLS:
            for bucket in sample_universe["duration_bucket"].unique():
                mask = sample_universe["duration_bucket"] == bucket
                vals = sample_universe.loc[mask, col]
                if len(vals) > 2:
                    assert abs(vals.std() - 1.0) < 0.3, f"{col} std not ~1 in {bucket}"

    def test_composite_is_weighted_sum(self, sample_universe):
        computed = sum(
            OPT_FACTOR_WEIGHTS[c] * sample_universe[c] for c in FACTOR_COLS
        )
        np.testing.assert_allclose(sample_universe["z_composite"], computed, atol=1e-4)

    def test_value_sign_convention(self, sample_universe):
        # Bonds with higher OAS (wider spread) should tend to have higher z_value
        median_oas = sample_universe["oas_bp"].median()
        wide = sample_universe["oas_bp"] > median_oas
        if wide.sum() > 5:
            assert sample_universe.loc[wide, "z_value"].mean() > 0

    def test_shock_application(self):
        df_base = build_universe(n=60)
        df_shocked = build_universe(n=60, shock=JUNE_SHOCK)
        # BOND015 should have higher oas_bp after shock (+15)
        b015_base = df_base[df_base["bond_id"] == "BOND015"]["oas_bp"].values[0]
        b015_shock = df_shocked[df_shocked["bond_id"] == "BOND015"]["oas_bp"].values[0]
        assert b015_shock > b015_base

    def test_deterministic_seed(self):
        df1 = build_universe(n=30, seed=99)
        df2 = build_universe(n=30, seed=99)
        pd.testing.assert_frame_equal(df1, df2)

    def test_oas_bounds(self, sample_universe):
        assert (sample_universe["oas_bp"] >= 20).all()
        assert (sample_universe["oas_bp"] <= 400).all()

    def test_different_seeds_differ(self):
        df1 = build_universe(n=30, seed=1)
        df2 = build_universe(n=30, seed=2)
        assert not df1["oas_bp"].equals(df2["oas_bp"])

    def test_small_universe(self):
        df = build_universe(n=10)
        assert len(df) == 10

    def test_has_mid_duration(self, sample_universe):
        assert "mid_duration" in sample_universe.columns
        assert (sample_universe["mid_duration"] > 0).all()
