"""Unit tests for _run_inline_bl() in the BL pipeline."""

import numpy as np
import pytest

from credit_portfolio.data.loader import load
from credit_portfolio.models.hmm_regime import fit_hmm
from credit_portfolio.pipelines.bl_pipeline import _run_inline_bl
from credit_portfolio.data.constants import OMEGA_SCALE


@pytest.fixture
def bl_pipeline_data(fred_csv_path):
    """Load FRED data and fit HMM for BL pipeline tests."""
    df = load(fred_csv_path)
    hmm_result = fit_hmm(df)
    return df, hmm_result


class TestRunInlineBL:
    def test_returns_dict_with_required_keys(self, bl_pipeline_data):
        df, hmm_result = bl_pipeline_data
        result = _run_inline_bl(df, hmm_result)
        for key in ("Pi", "mu_BL", "Q_vec", "Sigma", "omega_scale", "oas_chg_bp", "regime_info"):
            assert key in result, f"Missing key: {key}"

    def test_pi_shape_4(self, bl_pipeline_data):
        df, hmm_result = bl_pipeline_data
        result = _run_inline_bl(df, hmm_result)
        assert result["Pi"].shape == (4,)

    def test_sigma_4x4_positive_definite(self, bl_pipeline_data):
        df, hmm_result = bl_pipeline_data
        result = _run_inline_bl(df, hmm_result)
        Sigma = result["Sigma"]
        assert Sigma.shape == (4, 4)
        eigenvalues = np.linalg.eigvalsh(Sigma)
        assert all(eigenvalues > 0), f"Sigma not PD: eigenvalues={eigenvalues}"

    def test_mu_bl_all_finite(self, bl_pipeline_data):
        df, hmm_result = bl_pipeline_data
        result = _run_inline_bl(df, hmm_result)
        assert np.all(np.isfinite(result["mu_BL"]))

    def test_omega_scale_matches_regime(self, bl_pipeline_data):
        df, hmm_result = bl_pipeline_data
        result = _run_inline_bl(df, hmm_result)
        expected_scale = OMEGA_SCALE[hmm_result.current_regime]
        assert result["omega_scale"] == expected_scale
