"""Lock down all constant values so drift is caught immediately."""

import pytest
from credit_portfolio.data.constants import (
    SERIES_MAP, DURATIONS, MARKET_WEIGHTS, MARKET_WEIGHTS_WITH_HY,
    IG_MARKET_WEIGHTS, LAMBDA_RISK_AVERSION, TAU_BY_REGIME,
    REGIME_LABELS, REGIME_COLORS, OMEGA_SCALE, SECTORS, RATINGS,
    DURATION_BUCKETS, FRED_SERIES, STYLE,
    IG_ASSETS, IG_MARKET_WEIGHTS_ARRAY, ASSET_LABELS,
    PIPELINE_DELTA, PIPELINE_TAU, JUNE_SHOCK,
)


class TestSeriesMap:
    def test_has_18_entries(self):
        assert len(SERIES_MAP) == 18

    def test_oas_ig_mapping(self):
        assert SERIES_MAP["BAMLC0A0CM"] == "oas_ig"

    def test_all_values_are_strings(self):
        for k, v in SERIES_MAP.items():
            assert isinstance(k, str) and isinstance(v, str)


class TestDurations:
    def test_all_positive(self):
        for k, v in DURATIONS.items():
            assert v > 0, f"DURATIONS[{k!r}] = {v} is not positive"

    def test_ig_rating_tiers_present(self):
        for tier in ["oas_aaa", "oas_aa", "oas_a", "oas_bbb"]:
            assert tier in DURATIONS

    def test_aaa_duration_8_5(self):
        assert DURATIONS["oas_aaa"] == 8.5

    def test_bbb_duration_6_5(self):
        assert DURATIONS["oas_bbb"] == 6.5


class TestMarketWeights:
    def test_ig_sum_to_one(self):
        assert abs(sum(MARKET_WEIGHTS.values()) - 1.0) < 1e-10

    def test_ig_with_hy_sum_to_one(self):
        assert abs(sum(MARKET_WEIGHTS_WITH_HY.values()) - 1.0) < 1e-10

    def test_ig_oas_keyed_sum_to_one(self):
        assert abs(sum(IG_MARKET_WEIGHTS.values()) - 1.0) < 1e-10

    def test_bbb_is_largest_ig(self):
        assert MARKET_WEIGHTS["BBB"] == max(MARKET_WEIGHTS.values())

    def test_specific_ig_oas_values(self):
        assert IG_MARKET_WEIGHTS["oas_aaa"] == 0.03
        assert IG_MARKET_WEIGHTS["oas_aa"] == 0.10
        assert IG_MARKET_WEIGHTS["oas_a"] == 0.38
        assert IG_MARKET_WEIGHTS["oas_bbb"] == 0.49


class TestRegimeConstants:
    def test_tau_by_regime_ordering(self):
        assert TAU_BY_REGIME["COMPRESSION"] < TAU_BY_REGIME["NORMAL"] < TAU_BY_REGIME["STRESS"]

    def test_omega_scale_ordering(self):
        assert OMEGA_SCALE[0] < OMEGA_SCALE[1] < OMEGA_SCALE[2]

    def test_regime_labels_three_states(self):
        assert set(REGIME_LABELS.keys()) == {0, 1, 2}

    def test_regime_labels_upper_case(self):
        for label in REGIME_LABELS.values():
            assert label == label.upper()

    def test_regime_colors_three_states(self):
        assert set(REGIME_COLORS.keys()) == {0, 1, 2}

    def test_lambda_risk_aversion_positive(self):
        assert LAMBDA_RISK_AVERSION > 0


class TestEnumerations:
    def test_sectors_has_8(self):
        assert len(SECTORS) == 8

    def test_ratings_are_ig(self):
        assert RATINGS == ["AAA", "AA", "A", "BBB"]

    def test_duration_buckets_has_4(self):
        assert len(DURATION_BUCKETS) == 4

    def test_fred_series_subset_of_series_map_values(self):
        sm_values = set(SERIES_MAP.values())
        for friendly_name in FRED_SERIES.keys():
            assert friendly_name in sm_values


class TestPipelineConstants:
    def test_ig_assets_length_4(self):
        assert len(IG_ASSETS) == 4

    def test_ig_market_weights_array_sums_to_one(self):
        assert abs(float(IG_MARKET_WEIGHTS_ARRAY.sum()) - 1.0) < 1e-10

    def test_ig_market_weights_array_matches_dict(self):
        for i, asset in enumerate(IG_ASSETS):
            assert abs(IG_MARKET_WEIGHTS_ARRAY[i] - IG_MARKET_WEIGHTS[asset]) < 1e-10

    def test_asset_labels_covers_ig_assets(self):
        for asset in IG_ASSETS:
            assert asset in ASSET_LABELS

    def test_pipeline_delta_positive(self):
        assert PIPELINE_DELTA > 0

    def test_pipeline_tau_positive(self):
        assert PIPELINE_TAU > 0

    def test_june_shock_has_entries(self):
        assert len(JUNE_SHOCK) > 0
        for bond_id, shock in JUNE_SHOCK.items():
            assert bond_id.startswith("BOND")
            assert isinstance(shock, dict)


class TestStyle:
    def test_fig_dpi(self):
        assert STYLE["fig_dpi"] == 150

    def test_color_primary_is_hex(self):
        assert STYLE["color_primary"].startswith("#")
