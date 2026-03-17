"""
Monte Carlo simulation for credit portfolio risk analysis.

Two methods:
  - Empirical bootstrap: sample with replacement from historical returns.
  - Parametric normal: fit N(mu, sigma^2), then sample.

Outputs: simulated paths, VaR/CVaR at configurable confidence levels,
percentile bands for fan charts, terminal return distribution.
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class MonteCarloResult:
    """Full output of a Monte Carlo simulation."""
    method: str                           # "bootstrap" or "parametric"
    n_sims: int
    horizon_months: int
    simulated_paths: np.ndarray           # shape (n_sims, horizon_months)
    terminal_returns: np.ndarray          # shape (n_sims,) — cumulative at horizon
    percentile_bands: dict                # {pct: array of length horizon_months}
    median_path: np.ndarray               # median cumulative path
    risk_metrics: pd.DataFrame            # rows = confidence levels, cols = VaR/CVaR/P(loss)


def _compute_risk_metrics(terminal_returns: np.ndarray,
                          confidence_levels: list[float]) -> pd.DataFrame:
    """Compute VaR, CVaR, and P(loss) at each confidence level."""
    rows = []
    for cl in confidence_levels:
        alpha = 1 - cl / 100.0
        var_threshold = np.percentile(terminal_returns, alpha * 100)
        tail = terminal_returns[terminal_returns <= var_threshold]
        cvar = float(tail.mean()) if len(tail) > 0 else var_threshold
        p_loss = float((terminal_returns < 0).mean())

        rows.append({
            "Confidence": f"{cl:.0f}%",
            "VaR": var_threshold,
            "CVaR": cvar,
            "P(Loss)": p_loss,
        })
    return pd.DataFrame(rows)


def run_monte_carlo(
    historical_returns: pd.Series | np.ndarray,
    n_sims: int = 1000,
    horizon_months: int = 12,
    method: str = "bootstrap",
    confidence_levels: list[float] | None = None,
    seed: int = 42,
) -> MonteCarloResult:
    """
    Run Monte Carlo simulation on historical portfolio returns.

    Parameters
    ----------
    historical_returns : Monthly return series (decimal, e.g. 0.005 = 0.5%).
    n_sims : Number of simulation paths.
    horizon_months : Months to project forward.
    method : "bootstrap" (empirical) or "parametric" (normal).
    confidence_levels : List of confidence levels for VaR/CVaR (e.g. [90, 95, 99]).
    seed : Random seed.

    Returns
    -------
    MonteCarloResult with paths, risk metrics, and percentile bands.
    """
    if confidence_levels is None:
        confidence_levels = [90.0, 95.0, 99.0]

    rng = np.random.default_rng(seed)
    returns = np.asarray(historical_returns).flatten()
    returns = returns[np.isfinite(returns)]

    if len(returns) < 6:
        raise ValueError(f"Need at least 6 monthly returns, got {len(returns)}")

    # Generate simulated monthly returns
    if method == "bootstrap":
        # Sample with replacement from historical returns
        sim_returns = rng.choice(returns, size=(n_sims, horizon_months), replace=True)
    else:
        # Parametric normal
        mu = float(np.mean(returns))
        sigma = float(np.std(returns, ddof=1))
        sim_returns = rng.normal(mu, sigma, size=(n_sims, horizon_months))

    # Cumulative wealth paths: (1 + r1)(1 + r2)... - 1
    cumulative_paths = np.cumprod(1 + sim_returns, axis=1) - 1

    # Terminal returns
    terminal_returns = cumulative_paths[:, -1]

    # Percentile bands for fan chart
    percentiles = [5, 25, 50, 75, 95]
    bands = {}
    for p in percentiles:
        bands[p] = np.percentile(cumulative_paths, p, axis=0)

    median_path = bands[50]

    # Risk metrics
    risk_df = _compute_risk_metrics(terminal_returns, confidence_levels)

    return MonteCarloResult(
        method=method,
        n_sims=n_sims,
        horizon_months=horizon_months,
        simulated_paths=cumulative_paths,
        terminal_returns=terminal_returns,
        percentile_bands=bands,
        median_path=median_path,
        risk_metrics=risk_df,
    )
