"""Tests for models/prophet_views.py"""

import numpy as np
import pandas as pd
import pytest

from credit_portfolio.models.prophet_views import oas_to_expected_return


class TestOasToExpectedReturn:
    def test_carry_only_return(self):
        # delta=0 -> carry only
        ret = oas_to_expected_return(current_oas=1.0, delta_oas=0.0,
                                      duration=7.0, horizon_months=3)
        expected_carry = (1.0 / 100) * (3 / 12)
        assert abs(ret - expected_carry) < 1e-10

    def test_tightening_positive_price_return(self):
        # Tightening (negative delta) should add to return
        ret = oas_to_expected_return(current_oas=1.0, delta_oas=-0.10,
                                      duration=7.0, horizon_months=3)
        carry_only = oas_to_expected_return(current_oas=1.0, delta_oas=0.0,
                                             duration=7.0, horizon_months=3)
        assert ret > carry_only

    def test_known_value_manual(self):
        # BBB OAS=1.50%, tighten 15bp, dur=7.1, 3m
        ret = oas_to_expected_return(1.50, -0.15, 7.1, 3)
        carry = (1.50 / 100) * (3 / 12)
        price = 7.1 * (0.15 / 100) * (3 / 12)
        assert abs(ret - (carry + price)) < 1e-10

    def test_zero_oas_edge_case(self):
        ret = oas_to_expected_return(0.0, 0.0, 7.0, 3)
        assert ret == 0.0

    def test_view_dict_structure(self):
        # Just verify the function can return values used in view dicts
        ret = oas_to_expected_return(0.82, -0.05, 6.5, 3)
        assert isinstance(ret, float)
        assert np.isfinite(ret)

    def test_widening_negative_price_return(self):
        # Widening (positive delta) -> negative price return
        ret = oas_to_expected_return(1.0, 0.20, 7.0, 3)
        carry_only = oas_to_expected_return(1.0, 0.0, 7.0, 3)
        assert ret < carry_only

    @pytest.mark.slow
    def test_prophet_fit_returns_forecast(self, fred_csv_path):
        from credit_portfolio.data.loader import load
        from credit_portfolio.models.prophet_views import fit_prophet_for_bucket
        df = load(fred_csv_path)
        result = fit_prophet_for_bucket(df["oas_ig"].dropna(), horizon_months=3)
        assert "forecast_oas" in result
        assert "delta_oas" in result
        assert np.isfinite(result["forecast_oas"])
