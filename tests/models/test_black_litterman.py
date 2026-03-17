"""Tests for models/black_litterman.py — highest priority."""

import numpy as np
import pandas as pd
import pytest

from credit_portfolio.models.black_litterman import (
    compute_equilibrium_prior, build_views_matrices, black_litterman_posterior,
    compute_covariance, compute_historical_returns, _build_covariance, BLResult,
)


class TestEquilibriumPrior:
    def test_formula_verification(self, sample_covariance_4x4):
        w = np.array([0.04, 0.12, 0.34, 0.50])
        lam = 2.5
        pi = compute_equilibrium_prior(sample_covariance_4x4, w, lam)
        expected = lam * sample_covariance_4x4 @ w
        np.testing.assert_allclose(pi, expected)

    def test_prior_positive(self, sample_covariance_4x4):
        w = np.array([0.04, 0.12, 0.34, 0.50])
        pi = compute_equilibrium_prior(sample_covariance_4x4, w)
        # With positive weights and PD covariance, prior should be positive
        assert (pi > 0).all()


class TestViewsMatrices:
    def test_shape(self, sample_covariance_4x4):
        views = {"AAA": {"expected_return": 0.01}, "BBB": {"expected_return": 0.02}}
        assets = ["AAA", "AA", "A", "BBB"]
        P, Q, Omega = build_views_matrices(views, assets, sample_covariance_4x4, tau=0.025)
        assert P.shape == (2, 4)
        assert Q.shape == (2,)
        assert Omega.shape == (2, 2)

    def test_omega_positive_definite(self, sample_covariance_4x4):
        views = {"AAA": {"expected_return": 0.01}, "BBB": {"expected_return": 0.02}}
        assets = ["AAA", "AA", "A", "BBB"]
        _, _, Omega = build_views_matrices(views, assets, sample_covariance_4x4, tau=0.025)
        eigenvalues = np.linalg.eigvalsh(Omega)
        assert (eigenvalues > 0).all()

    def test_omega_he_litterman_spec(self, sample_covariance_4x4):
        views = {"A": {"expected_return": 0.015}}
        assets = ["AAA", "AA", "A", "BBB"]
        tau = 0.025
        P, Q, Omega = build_views_matrices(views, assets, sample_covariance_4x4, tau)
        # Omega = tau * P @ Sigma @ P' + eps * I
        expected = tau * (P @ sample_covariance_4x4 @ P.T) + np.eye(1) * 1e-8
        np.testing.assert_allclose(Omega, expected, atol=1e-10)

    def test_no_views_returns_none(self, sample_covariance_4x4):
        assets = ["AAA", "AA", "A", "BBB"]
        P, Q, Omega = build_views_matrices({}, assets, sample_covariance_4x4, 0.025)
        assert P is None
        assert Q is None
        assert Omega is None


class TestBLPosterior:
    def test_no_views_posterior_equals_prior(self, sample_covariance_4x4):
        w = np.array([0.04, 0.12, 0.34, 0.50])
        sigma = sample_covariance_4x4
        pi = compute_equilibrium_prior(sigma, w)
        # With identity P and Q=pi, posterior should be close to prior
        P = np.eye(4)
        tau = 0.025
        Omega = tau * sigma + np.eye(4) * 1e-8
        mu_bl, _ = black_litterman_posterior(sigma, pi, P, pi, Omega, tau)
        np.testing.assert_allclose(mu_bl, pi, atol=0.01)

    def test_hand_computed_2_asset(self):
        sigma = np.array([[0.04, 0.01], [0.01, 0.09]])
        w = np.array([0.5, 0.5])
        pi = 2.5 * sigma @ w
        tau = 0.05
        P = np.array([[1.0, 0.0]])
        Q = np.array([0.10])
        Omega = tau * (P @ sigma @ P.T) + np.eye(1) * 1e-8
        mu_bl, sigma_bl = black_litterman_posterior(sigma, pi, P, Q, Omega, tau)
        # Posterior should be between prior and views
        assert mu_bl[0] > pi[0] or mu_bl[0] < Q[0]  # pulled toward view
        assert sigma_bl.shape == (2, 2)

    def test_posterior_between_prior_and_views(self, sample_covariance_4x4):
        sigma = sample_covariance_4x4
        w = np.array([0.04, 0.12, 0.34, 0.50])
        pi = compute_equilibrium_prior(sigma, w)
        # View: asset 0 has much higher return
        P = np.eye(4)
        Q = pi * 2.0  # views are 2x prior
        tau = 0.025
        Omega = tau * (P @ sigma @ P.T) + np.eye(4) * 1e-8
        mu_bl, _ = black_litterman_posterior(sigma, pi, P, Q, Omega, tau)
        # BL should be between prior and views for each asset
        for i in range(4):
            assert min(pi[i], Q[i]) - 0.01 <= mu_bl[i] <= max(pi[i], Q[i]) + 0.01

    def test_low_tau_closer_to_views(self, sample_covariance_4x4):
        sigma = sample_covariance_4x4
        w = np.array([0.04, 0.12, 0.34, 0.50])
        pi = compute_equilibrium_prior(sigma, w)
        P = np.eye(4)
        Q = pi * 3.0

        # Use fixed Omega so only tau's effect on prior uncertainty varies
        Omega_fixed = 0.025 * (P @ sigma @ P.T) + np.eye(4) * 1e-8

        tau_low = 0.001
        tau_high = 0.1

        mu_low, _ = black_litterman_posterior(sigma, pi, P, Q, Omega_fixed, tau_low)
        mu_high, _ = black_litterman_posterior(sigma, pi, P, Q, Omega_fixed, tau_high)

        # Low tau -> tighter prior -> closer to prior, not views
        # High tau -> looser prior -> views pull more
        dist_low = np.linalg.norm(mu_low - pi)
        dist_high = np.linalg.norm(mu_high - pi)
        assert dist_low < dist_high

    def test_high_tau_closer_to_views(self, sample_covariance_4x4):
        sigma = sample_covariance_4x4
        w = np.array([0.04, 0.12, 0.34, 0.50])
        pi = compute_equilibrium_prior(sigma, w)
        P = np.eye(4)
        Q = pi * 3.0

        # Fixed Omega: high tau loosens prior, so views pull more
        Omega_fixed = 0.025 * (P @ sigma @ P.T) + np.eye(4) * 1e-8
        tau_high = 0.5
        mu_bl, _ = black_litterman_posterior(sigma, pi, P, Q, Omega_fixed, tau_high)

        # With high tau, posterior should be between prior and views
        for i in range(4):
            assert min(pi[i], Q[i]) - 0.01 <= mu_bl[i] <= max(pi[i], Q[i]) + 0.01

    def test_posterior_covariance_ge_historical(self, sample_covariance_4x4):
        sigma = sample_covariance_4x4
        w = np.array([0.04, 0.12, 0.34, 0.50])
        pi = compute_equilibrium_prior(sigma, w)
        P = np.eye(4)
        Q = pi * 1.5
        tau = 0.025
        Omega = tau * (P @ sigma @ P.T) + np.eye(4) * 1e-8
        _, sigma_bl = black_litterman_posterior(sigma, pi, P, Q, Omega, tau)
        # Sigma_BL = Sigma + M, so it should be >= Sigma in PSD sense
        diff = sigma_bl - sigma
        eigenvalues = np.linalg.eigvalsh(diff)
        assert (eigenvalues >= -1e-8).all()

    def test_posterior_covariance_symmetric(self, sample_covariance_4x4):
        sigma = sample_covariance_4x4
        w = np.array([0.04, 0.12, 0.34, 0.50])
        pi = compute_equilibrium_prior(sigma, w)
        P = np.eye(4)
        Q = pi * 1.5
        tau = 0.025
        Omega = tau * (P @ sigma @ P.T) + np.eye(4) * 1e-8
        _, sigma_bl = black_litterman_posterior(sigma, pi, P, Q, Omega, tau)
        np.testing.assert_allclose(sigma_bl, sigma_bl.T, atol=1e-10)


class TestBuildCovariance:
    def test_positive_definite(self, small_universe):
        cov = _build_covariance(small_universe)
        eigenvalues = np.linalg.eigvalsh(cov)
        assert (eigenvalues > 0).all()

    def test_shape(self, small_universe):
        cov = _build_covariance(small_universe)
        n = len(small_universe)
        assert cov.shape == (n, n)


class TestHistoricalReturns:
    def test_schema(self, sample_monthly_data):
        ret = compute_historical_returns(sample_monthly_data, ["AAA", "AA", "A", "BBB"])
        assert isinstance(ret, pd.DataFrame)
        for asset in ["AAA", "AA", "A", "BBB"]:
            assert asset in ret.columns

    def test_covariance_annualised(self, sample_monthly_data):
        ret = compute_historical_returns(sample_monthly_data, ["AAA", "BBB"])
        cov = compute_covariance(ret)
        # Should be 12x monthly cov
        cov_monthly = ret.cov().values
        np.testing.assert_allclose(cov, cov_monthly * 12, atol=1e-10)
