"""Tests for data/loader.py"""

import numpy as np
import pandas as pd
import pytest

from credit_portfolio.data.loader import load, monthly_returns, oas_changes, compute_analytics


class TestLoad:
    def test_returns_dataframe(self, fred_csv_path):
        df = load(fred_csv_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 100

    def test_correct_columns(self, fred_csv_path):
        df = load(fred_csv_path)
        assert "oas_ig" in df.columns

    def test_date_range_filtering(self, fred_csv_path):
        df = load(fred_csv_path, start="2010-01-01")
        assert df.index[0] >= pd.Timestamp("2010-01-01")

    def test_month_end_resampling(self, fred_csv_path):
        df = load(fred_csv_path, freq="ME")
        # All dates should be month-end
        for dt in df.index[:10]:
            assert dt == dt + pd.offsets.MonthEnd(0)

    def test_column_renaming(self, fred_csv_path):
        df = load(fred_csv_path)
        # Should have friendly names, not FRED IDs
        assert "BAMLC0A0CM" not in df.columns
        assert "oas_ig" in df.columns

    def test_forward_fill(self, fred_csv_path):
        df = load(fred_csv_path)
        # oas_ig should have no NaN after load
        assert df["oas_ig"].isna().sum() == 0


class TestOasChanges:
    def test_horizons(self, fred_csv_path):
        df = load(fred_csv_path)
        changes = oas_changes(df)
        assert "oas_ig_chg1m" in changes.columns
        assert "oas_ig_chg3m" in changes.columns
        assert "oas_ig_chg6m" in changes.columns
        assert "oas_ig_chg12m" in changes.columns


class TestMonthlyReturns:
    def test_returns_series(self, fred_csv_path):
        df = load(fred_csv_path)
        ret = monthly_returns(df)
        assert isinstance(ret, pd.Series)
        assert len(ret) == len(df)


class TestComputeAnalytics:
    def test_keys(self, fred_csv_path):
        df = load(fred_csv_path)
        analytics = compute_analytics(df)
        assert "value_signal" in analytics
        assert "momentum_signal" in analytics
        assert "sharpe_ig" in analytics
        assert "raw" in analytics
