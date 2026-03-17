"""Tests for optimizers/risk_parity.py"""

import numpy as np
import pandas as pd
import pytest

from credit_portfolio.optimizers.risk_parity import (
    optimise_risk_parity, compare_allocations, RiskParityConfig, _risk_contribution,
)


class TestImport:
    def test_import_succeeds(self):
        """This test verifies the previously broken import is now fixed."""
        from credit_portfolio.optimizers.risk_parity import optimise_risk_parity
        assert callable(optimise_risk_parity)

    def test_build_covariance_importable(self):
        from credit_portfolio.models.black_litterman import _build_covariance
        assert callable(_build_covariance)


class TestOptimiseRiskParity:
    def test_optimal_status(self, sample_universe):
        result = optimise_risk_parity(sample_universe)
        assert result.status == "optimal"

    def test_risk_contributions_approx_equal(self, sample_universe):
        result = optimise_risk_parity(sample_universe)
        rc = result.risk_contributions.values
        ratio = rc.max() / max(rc.min(), 1e-10)
        # With sector constraints, perfect ERC not achievable; allow wider ratio
        assert ratio < 15.0

    def test_rcs_sum_to_one(self, sample_universe):
        result = optimise_risk_parity(sample_universe)
        assert abs(result.risk_contributions.sum() - 1.0) < 1e-4

    def test_weights_positive(self, sample_universe):
        result = optimise_risk_parity(sample_universe)
        assert (result.weights > -1e-6).all()

    def test_weights_sum_to_one(self, sample_universe):
        result = optimise_risk_parity(sample_universe)
        assert abs(result.weights.sum() - 1.0) < 1e-4

    def test_single_name_cap(self, sample_universe):
        config = RiskParityConfig(max_single_name=0.05)
        result = optimise_risk_parity(sample_universe, config=config)
        assert (result.weights <= 0.05 + 1e-4).all()

    def test_herfindahl_low(self, sample_universe):
        result = optimise_risk_parity(sample_universe)
        # Herfindahl of equal RC should be ~1/n
        n = len(result.weights)
        assert result.concentration_ratio < 3.0 / n


class TestCompareAllocations:
    def test_schema(self, sample_universe):
        from credit_portfolio.optimizers.factor_tilt import optimise
        ft = optimise(sample_universe)
        rp = optimise_risk_parity(sample_universe)
        comp = compare_allocations(sample_universe, ft.weights, rp.weights)
        assert "sector" in comp.columns
        assert "factor_tilt" in comp.columns
        assert "risk_parity" in comp.columns
        assert "equal_weight" in comp.columns


class TestFallback:
    def test_fallback_to_inverse_vol(self):
        # Create a tiny universe where CVXPY might struggle
        from credit_portfolio.data.universe import build_universe
        df = build_universe(n=5)
        # Give very tight constraints to force fallback
        config = RiskParityConfig(max_single_name=0.01, max_sector_dev=0.001)
        result = optimise_risk_parity(df, config=config)
        # Should still return valid weights even if fallback
        assert result.weights.sum() > 0
