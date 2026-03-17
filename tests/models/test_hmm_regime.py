"""Tests for models/hmm_regime.py"""

import numpy as np
import pandas as pd
import pytest

from credit_portfolio.models.hmm_regime import fit_hmm, get_current_regime, regime_summary


@pytest.fixture
def hmm_data(fred_csv_path):
    from credit_portfolio.data.loader import load
    return load(fred_csv_path)


@pytest.fixture
def hmm_result(hmm_data):
    return fit_hmm(hmm_data)


class TestFitHMM:
    def test_returns_hmm_result(self, hmm_result):
        from credit_portfolio.models.hmm_regime import HMMResult
        assert isinstance(hmm_result, HMMResult)

    def test_three_unique_states(self, hmm_result):
        assert len(hmm_result.states.unique()) == 3

    def test_state_ordering_by_mean_oas(self, hmm_result):
        # State 0 (compression) should have lower mean OAS than state 2 (stress)
        stats = hmm_result.regime_stats
        assert stats.loc["COMPRESSION", "mean_oas_bp"] < stats.loc["STRESS", "mean_oas_bp"]

    def test_regime_labels_correct(self, hmm_result):
        labels = set(hmm_result.regime_stats.index)
        assert labels == {"COMPRESSION", "NORMAL", "STRESS"}

    def test_tau_ordering(self, hmm_result):
        from credit_portfolio.data.constants import TAU_BY_REGIME
        assert TAU_BY_REGIME["COMPRESSION"] < TAU_BY_REGIME["NORMAL"] < TAU_BY_REGIME["STRESS"]

    def test_transition_matrix_rows_sum_to_one(self, hmm_result):
        T = hmm_result.transition_matrix
        for i in range(3):
            assert abs(T[i].sum() - 1.0) < 1e-4

    def test_transition_matrix_non_negative(self, hmm_result):
        assert (hmm_result.transition_matrix >= -1e-10).all()

    def test_state_probs_sum_to_one(self, hmm_result):
        probs = hmm_result.state_probs
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-4)

    def test_current_regime_is_last_state(self, hmm_result):
        last_state = hmm_result.states.iloc[-1]
        assert hmm_result.current_regime == last_state

    def test_regime_summary_month_counts(self, hmm_result):
        stats = hmm_result.regime_stats
        total = stats["n_months"].sum()
        assert total == len(hmm_result.states)
