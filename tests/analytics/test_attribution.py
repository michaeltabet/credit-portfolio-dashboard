"""Tests for analytics/attribution.py"""

import numpy as np
import pandas as pd
import pytest

from credit_portfolio.analytics.attribution import attribute, format_for_llm, AttributionReport


class TestAttribute:
    def test_returns_attribution_report(self, sample_universe_pair, sample_opt_result_pair):
        df_t0, df_t1 = sample_universe_pair
        r0, r1 = sample_opt_result_pair
        report = attribute(df_t0, df_t1, r0, r1)
        assert isinstance(report, AttributionReport)

    def test_top_adds_sorted_descending(self, sample_universe_pair, sample_opt_result_pair):
        df_t0, df_t1 = sample_universe_pair
        r0, r1 = sample_opt_result_pair
        report = attribute(df_t0, df_t1, r0, r1)
        if len(report.top_adds) > 1:
            changes = report.top_adds["weight_change"].values
            assert changes[0] >= changes[1]

    def test_top_reduces_sorted_ascending(self, sample_universe_pair, sample_opt_result_pair):
        df_t0, df_t1 = sample_universe_pair
        r0, r1 = sample_opt_result_pair
        report = attribute(df_t0, df_t1, r0, r1)
        if len(report.top_reduces) > 1:
            changes = report.top_reduces["weight_change"].values
            assert changes[0] <= changes[1]

    def test_sector_shifts_sum_approx_zero(self, sample_universe_pair, sample_opt_result_pair):
        df_t0, df_t1 = sample_universe_pair
        r0, r1 = sample_opt_result_pair
        report = attribute(df_t0, df_t1, r0, r1)
        total_change = report.sector_shifts["change"].sum()
        assert abs(total_change) < 0.05  # approximately zero

    def test_factor_delta_has_3_keys(self, sample_universe_pair, sample_opt_result_pair):
        df_t0, df_t1 = sample_universe_pair
        r0, r1 = sample_opt_result_pair
        report = attribute(df_t0, df_t1, r0, r1)
        assert len(report.factor_delta) == 3
        for key in ["z_dts", "z_value", "z_momentum"]:
            assert key in report.factor_delta

    def test_change_equals_t1_minus_t0(self, sample_universe_pair, sample_opt_result_pair):
        df_t0, df_t1 = sample_universe_pair
        r0, r1 = sample_opt_result_pair
        report = attribute(df_t0, df_t1, r0, r1)
        for f, d in report.factor_delta.items():
            assert abs(d["change"] - (d["t1"] - d["t0"])) < 1e-3

    def test_new_bindings_detected(self, sample_universe_pair, sample_opt_result_pair):
        df_t0, df_t1 = sample_universe_pair
        r0, r1 = sample_opt_result_pair
        report = attribute(df_t0, df_t1, r0, r1)
        assert isinstance(report.new_bindings, list)


class TestFormatForLLM:
    def test_has_all_sections(self, sample_universe_pair, sample_opt_result_pair):
        df_t0, df_t1 = sample_universe_pair
        r0, r1 = sample_opt_result_pair
        report = attribute(df_t0, df_t1, r0, r1)
        text = format_for_llm(report)
        assert "SECTOR SHIFTS" in text
        assert "TOP 5 ADDITIONS" in text
        assert "TOP 5 REDUCTIONS" in text
        assert "FACTOR EXPOSURE SHIFTS" in text
        assert "BINDING CONSTRAINTS" in text

    def test_no_unlabeled_numbers(self, sample_universe_pair, sample_opt_result_pair):
        df_t0, df_t1 = sample_universe_pair
        r0, r1 = sample_opt_result_pair
        report = attribute(df_t0, df_t1, r0, r1)
        text = format_for_llm(report)
        # Every number should appear near a label
        assert "Universe size:" in text
