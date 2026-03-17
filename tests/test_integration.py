"""Integration tests."""

import subprocess
import sys
import pytest

from credit_portfolio.cli import main as cli_main


class TestFactorPipelineE2E:
    def test_full_factor_pipeline(self, tmp_output_dir):
        from credit_portfolio.data.universe import build_universe, JUNE_SHOCK
        from credit_portfolio.optimizers.factor_tilt import optimise
        from credit_portfolio.analytics.attribution import attribute, format_for_llm
        from credit_portfolio.analytics.commentary import generate_commentary_mock

        df_t0 = build_universe(n=60)
        df_t1 = build_universe(n=60, shock=JUNE_SHOCK)
        r0 = optimise(df_t0)
        r1 = optimise(df_t1, prior_w=r0.weights.values)
        report = attribute(df_t0, df_t1, r0, r1)
        text = format_for_llm(report)
        commentary = generate_commentary_mock(report)
        assert len(text) > 100
        assert len(commentary) > 100


class TestRiskParityWithBLCov:
    def test_risk_parity_with_bl_covariance(self):
        from credit_portfolio.data.universe import build_universe
        from credit_portfolio.models.black_litterman import _build_covariance
        from credit_portfolio.optimizers.risk_parity import optimise_risk_parity

        df = build_universe(n=30)
        sigma = _build_covariance(df)
        result = optimise_risk_parity(df, sigma=sigma)
        assert result.status in ["optimal", "fallback_inv_vol"]
        assert abs(result.weights.sum() - 1.0) < 1e-4


@pytest.mark.slow
class TestBLPipelineE2E:
    def test_full_bl_pipeline(self, fred_csv_path, tmp_output_dir):
        from credit_portfolio.data.loader import load
        from credit_portfolio.models.hmm_regime import fit_hmm, get_current_regime

        df = load(fred_csv_path)
        hmm_result = fit_hmm(df)
        regime_info = get_current_regime(hmm_result)
        assert regime_info["regime"] in ["COMPRESSION", "NORMAL", "STRESS"]


class TestCLIParsing:
    def test_no_command_exits_1(self):
        with pytest.raises(SystemExit) as exc_info:
            cli_main([])
        assert exc_info.value.code == 1

    def test_factor_accepts_output_flag(self, tmp_output_dir):
        # Just verify the parser doesn't reject the flag (actual run tested below)
        result = subprocess.run(
            [sys.executable, "-m", "credit_portfolio", "factor",
             "--output", tmp_output_dir],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0

    def test_charts_subcommand_exists(self):
        # Should fail because no data, but should NOT fail on argument parsing
        result = subprocess.run(
            [sys.executable, "-m", "credit_portfolio", "charts",
             "--output", "/tmp/test_charts"],
            capture_output=True, text=True, timeout=30,
        )
        # returncode may be non-zero if data missing, but stderr should not
        # contain "unrecognized arguments" or "invalid choice"
        assert "invalid choice" not in result.stderr
        assert "unrecognized arguments" not in result.stderr


class TestCLI:
    def test_cli_factor_subcommand(self, tmp_output_dir):
        result = subprocess.run(
            [sys.executable, "-m", "credit_portfolio", "factor",
             "--output", tmp_output_dir],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0

    @pytest.mark.slow
    def test_cli_bl_subcommand(self, tmp_output_dir):
        result = subprocess.run(
            [sys.executable, "-m", "credit_portfolio", "bl",
             "--output", tmp_output_dir],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0
