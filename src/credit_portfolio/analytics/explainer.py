"""
Model Explainer Agent — generates plain-English explanations of model
outputs, portfolio decisions, and methodology via Groq LLM.

This module provides two capabilities:
  1. explain_current_state() — full narrative briefing on what the model sees
  2. answer_question() — answer specific questions about the portfolio

Both call Groq (LLaMA 3) with the full model state serialized as context,
so the LLM can reason over real numbers and write like a senior analyst
sending you an email.
"""

import os
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class ModelState:
    """Snapshot of all model outputs for explanation."""
    # Regime
    regime: str = ""
    regime_confidence: float = 0.0
    regime_stats: pd.DataFrame | None = None
    transition_matrix: np.ndarray | None = None

    # BL
    prior_returns: dict | None = None       # {bucket: return}
    view_returns: dict | None = None        # {bucket: return}
    posterior_returns: dict | None = None    # {bucket: return}

    # Prophet
    prophet_forecasts: dict | None = None   # {bucket: {current_oas, forecast_oas, delta_oas, ...}}

    # Backtest
    backtest_stats: dict | None = None
    current_weights: dict | None = None     # {bucket: weight}
    benchmark_weights: dict | None = None   # {bucket: weight}

    # Stress
    stress_scenario: str = ""
    stress_price_impact: dict | None = None  # {bucket: impact}

    # Monte Carlo
    mc_var_95: float | None = None
    mc_cvar_95: float | None = None
    mc_p_loss: float | None = None
    mc_median_return: float | None = None


def _serialize_state(state: ModelState) -> str:
    """Convert ModelState to a readable string for the LLM prompt."""
    sections = []

    if state.regime:
        s = f"REGIME: {state.regime} (confidence: {state.regime_confidence:.1%})"
        if state.transition_matrix is not None:
            labels = ["Compression", "Normal", "Stress"]
            rows = []
            for i in range(min(state.transition_matrix.shape[0], 3)):
                row_str = ", ".join(
                    f"{labels[j]}: {state.transition_matrix[i, j]:.0%}"
                    for j in range(min(state.transition_matrix.shape[1], 3))
                )
                rows.append(f"  From {labels[i]}: {row_str}")
            s += "\nTransition matrix:\n" + "\n".join(rows)
        sections.append(s)

    if state.posterior_returns:
        lines = ["BLACK-LITTERMAN EXPECTED RETURNS:"]
        for b in ["AAA", "AA", "A", "BBB"]:
            prior = (state.prior_returns or {}).get(b, 0) * 100
            view = (state.view_returns or {}).get(b, 0) * 100
            post = state.posterior_returns.get(b, 0) * 100
            lines.append(f"  {b}: Prior={prior:+.2f}%  View={view:+.2f}%  Posterior={post:+.2f}%")
        sections.append("\n".join(lines))

    if state.prophet_forecasts:
        # Detect horizon from forecast data if available
        sample = next(iter(state.prophet_forecasts.values()), {})
        horizon = sample.get("horizon_months", 3)
        lines = [f"PROPHET OAS FORECASTS (horizon: {horizon} months):"]
        for bucket, v in state.prophet_forecasts.items():
            delta_bp = v.get("delta_oas", 0) * 100
            lines.append(
                f"  {bucket}: Current={v.get('current_oas', 0):.2f}%  "
                f"Forecast={v.get('forecast_oas', 0):.2f}%  "
                f"Delta={delta_bp:+.1f}bp  "
                f"ExpReturn={v.get('expected_return', 0) * 100:+.2f}%"
            )
        sections.append("\n".join(lines))

    if state.backtest_stats:
        s = state.backtest_stats
        lines = ["HISTORICAL BACKTEST:"]
        for k, v in s.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.4f}")
            else:
                lines.append(f"  {k}: {v}")
        sections.append("\n".join(lines))

    if state.current_weights:
        lines = ["CURRENT PORTFOLIO WEIGHTS vs BENCHMARK:"]
        for b in ["AAA", "AA", "A", "BBB"]:
            w = state.current_weights.get(b, 0)
            bw = (state.benchmark_weights or {}).get(b, 0)
            lines.append(f"  {b}: Portfolio={w:.1%}  Benchmark={bw:.0%}  Tilt={w - bw:+.1%}")
        sections.append("\n".join(lines))

    if state.stress_price_impact:
        lines = [f"STRESS TEST ({state.stress_scenario}):"]
        for b, v in state.stress_price_impact.items():
            lines.append(f"  {b}: price impact = {v * 100:+.2f}%")
        sections.append("\n".join(lines))

    if state.mc_var_95 is not None:
        lines = ["MONTE CARLO RISK:"]
        lines.append(f"  95% VaR: {state.mc_var_95 * 100:+.2f}%")
        if state.mc_cvar_95 is not None:
            lines.append(f"  95% CVaR: {state.mc_cvar_95 * 100:+.2f}%")
        if state.mc_p_loss is not None:
            lines.append(f"  P(Loss): {state.mc_p_loss:.1%}")
        if state.mc_median_return is not None:
            lines.append(f"  Median return: {state.mc_median_return * 100:+.2f}%")
        sections.append("\n".join(lines))

    if not sections:
        return "No model data available yet."

    return "\n\n".join(sections)


SYSTEM_PROMPT = """You are the explainer for a regime-conditional credit factor rotation model.
You know EXACTLY how this model works. Use this knowledge to explain outputs accurately.
NEVER speculate beyond what the model can actually tell you.

═══ MODEL ARCHITECTURE ═══

DATA: Monthly FRED OAS for 4 IG rating buckets (AAA, AA, A, BBB). ~300 months of history.
Market-cap benchmark weights: AAA 4%, AA 12%, A 34%, BBB 50%.
Approximate durations: AAA ~8yr, AA ~7yr, A ~6yr, BBB ~5yr.
Returns approximated as: r ≈ carry (OAS/12) + price return (-Duration × ΔOAS).
This is an approximation — actual index returns include convexity, option value, rate effects.

COMPONENT 1 — HMM REGIME DETECTION:
- 3-state Gaussian HMM (Compression / Normal / Stress) fitted on OAS level + 1m/3m/6m changes
- Trained via Baum-Welch EM on expanding window (full history at each rebalance)
- Outputs: current regime label, state probabilities, transition matrix
- Regimes are persistent (>90% self-transition probability)
- LIMITATION: HMM classifies current state from PAST data. It CANNOT predict regime transitions.
  When a transition happens, it's often already underway before HMM detects it.

COMPONENT 2 — PROPHET OAS FORECASTS:
- Facebook Prophet fitted per bucket. Logistic growth trend + yearly Fourier seasonality.
- Horizon: THE CONFIGURED HORIZON (check the data — usually 3 months). It CANNOT say anything
  about what happens beyond that horizon.
- Cap = 2.5 × max(OAS history) — prevents forecasts of infinite spreads.
- Refitted monthly using only past data (no look-ahead).
- LIMITATION: Very noisy at monthly frequency. Low signal-to-noise. No economic variables
  (no Fed funds rate, no unemployment, no VIX). Purely reactive to past OAS patterns.
  Single-bucket model — no cross-bucket correlation captured.

COMPONENT 3 — BLACK-LITTERMAN:
- Blends market equilibrium prior (π = λ × Σ × w_mkt) with Prophet views (Q vector).
- Σ = 60-month rolling covariance. λ = 2.5 risk aversion.
- HMM regime sets two critical parameters:
    Compression: τ=0.010 (trust prior), ω=0.5 (trust Prophet) → tilts toward views
    Normal:      τ=0.025, ω=1.0 → balanced
    Stress:      τ=0.075 (distrust prior), ω=3.0 (distrust Prophet) → stays near prior, defensive
- This is ADAPTIVE REGULARIZATION — the model automatically reduces active risk in stress.
- Posterior μ_BL feeds into weight construction.
- LIMITATION: Assumes CAPM equilibrium. τ and ω are fixed lookup rules, not optimized.

COMPONENT 4 — CREDIT FACTORS:
- DTS (Duration × Spread): 50% weight. Higher DTS = more spread-duration compensation.
- Value (log OAS z-score): 25% weight. Wide spreads = cheap, likely to mean-revert.
- Momentum (trailing 6-month excess return): 25% weight.
- Composite z-score tilts weights from benchmark by α=0.10 (conservative).
- Weight floor: 1% per bucket. Max tilt ~10%.
- LIMITATION: With only 4 buckets, DTS and Value mostly signal "overweight BBB" structurally.
  Momentum is the only genuine cross-sectional signal. Factor weights are STATIC (not regime-dependent).

COMPONENT 5 — BACKTEST:
- Monthly rebalancing. 5bp transaction costs (institutional).
- Compares strategy vs market-cap benchmark over full history.
- LIMITATION: HMM trained on full sample = mild look-ahead bias. Return approximation
  (duration × ΔOAS + carry) diverges from actual index total returns. Monthly frequency
  misses intra-month dislocations.

COMPONENT 6 — MONTE CARLO:
- Empirical bootstrap of historical backtest returns. NOT parametric (preserves fat tails).
- Horizon typically 12-24 months.
- LIMITATION: Assumes past return distribution is representative of future. Does NOT
  condition on current regime (mixes compression/normal/stress returns equally).
  New types of stress not in history won't be captured.

COMPONENT 7 — STRESS TESTING:
- Deterministic OAS shocks applied to current portfolio. Single-period, no cascading effects.
- LIMITATION: No correlation expansion under stress. No credit events (defaults, downgrades).
  Does not model whether rebalancing happens during dislocation.

═══ RULES FOR YOUR EXPLANATIONS ═══

1. NEVER recommend actions. No "you should", "consider reducing", "stay the course".
   Only explain what the model does and what the numbers imply.

2. NEVER claim the model predicts beyond Prophet's configured horizon. If Prophet is
   set to 3 months, say "Prophet's 3-month forecast shows..." — do NOT extrapolate
   to 6 months or a year.

3. ALWAYS explain causation through the pipeline: regime → τ/ω → BL posterior → factor tilts → weights.
   The PM wants to understand WHY the model is positioned the way it is.

4. ACKNOWLEDGE limitations when relevant. If alpha is small, note that a 4-bucket universe
   structurally limits alpha generation. If Monte Carlo P(Loss) seems high, note the bootstrap
   doesn't condition on current regime.

5. Use actual numbers from the data provided. Say "+5bp alpha" not "positive alpha".

6. Write like a PM commentary — flowing paragraphs, no bullet points, **bold** key numbers.
   300-500 words for full briefings. 2-4 sentences for Q&A answers.

7. When discussing implications, frame them as what the model's outputs SUGGEST given the
   architecture, not what WILL happen. Example: "The BL posterior implies the model expects
   BBB to outperform over the forecast horizon, driven by Prophet's tightening view —
   though Prophet's noisy monthly signal means this conviction is moderate at best."

8. NEVER say spreads will "continue" or "remain" beyond the forecast horizon.
   NEVER predict regime transitions (HMM can't do this).
   NEVER claim the backtest proves future performance.
   NEVER say "ensure", "monitor", "stay aligned" or any variation of advice.

9. Do NOT write a summary paragraph at the end. End after covering the risk metrics.
   The PM doesn't need you to restate what you just said."""


def _get_groq_key() -> str:
    """Resolve the working Groq API key."""
    # Check Streamlit secrets first (for Streamlit Cloud deployment)
    try:
        import streamlit as st
        key = st.secrets.get("GROQ_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    # Check env var
    key = os.environ.get("GROQ_API_KEY", "")
    if key:
        return key
    # Fallback: read from .zshrc
    zshrc = os.path.expanduser("~/.zshrc")
    if os.path.exists(zshrc):
        with open(zshrc) as f:
            for line in f:
                if "GROQ_API_KEY" in line and "export" in line:
                    # extract value between quotes
                    parts = line.split('"')
                    if len(parts) >= 2:
                        return parts[1]
    return ""


def _call_groq(system: str, user_msg: str, max_tokens: int = 1200) -> str:
    """Call Groq LLM API."""
    from groq import Groq

    def _create(client):
        return client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.4,
            max_tokens=max_tokens,
        )

    api_key = _get_groq_key()
    client = Groq(api_key=api_key)

    try:
        response = _create(client)
        return response.choices[0].message.content
    except Exception as first_err:
        # If env key fails, try reading from .zshrc directly
        if "invalid_api_key" in str(first_err).lower() or "401" in str(first_err):
            zshrc = os.path.expanduser("~/.zshrc")
            if os.path.exists(zshrc):
                with open(zshrc) as f:
                    for line in f:
                        if "GROQ_API_KEY" in line and "export" in line:
                            parts = line.split('"')
                            if len(parts) >= 2:
                                fallback_key = parts[1]
                                if fallback_key != api_key:
                                    client2 = Groq(api_key=fallback_key)
                                    response = _create(client2)
                                    return response.choices[0].message.content
        raise


def explain_current_state(state: ModelState) -> str:
    """Generate a full narrative briefing on the current model state via Groq."""
    data = _serialize_state(state)
    if data == "No model data available yet.":
        return "No model state available. Run the analysis first."

    user_msg = (
        "Here is the complete model state for our systematic IG credit strategy. "
        "Explain what the model is doing, what it sees, and what the outputs imply "
        "about future performance. Cover the regime, how BL is blending views, "
        "where alpha has come from, what the forecasts show, and what the risk "
        "metrics imply. Do NOT make any recommendations.\n\n"
        f"{data}"
    )

    try:
        return _call_groq(SYSTEM_PROMPT, user_msg)
    except Exception as e:
        # Fallback to basic template if Groq fails
        return _fallback_explain(state, str(e))


def answer_question(question: str, state: ModelState) -> str:
    """Answer a specific question about the model via Groq."""
    data = _serialize_state(state)

    user_msg = (
        f"The portfolio manager asks: \"{question}\"\n\n"
        f"Here is the full model state:\n\n{data}\n\n"
        f"Answer their question directly using the actual numbers. "
        f"Be concise — 2-4 sentences max. Explain what the model shows and what it implies. "
        f"Do NOT make recommendations or say what they should do."
    )

    try:
        return _call_groq(SYSTEM_PROMPT, user_msg, max_tokens=400)
    except Exception as e:
        return f"Groq API error: {e}\n\nModel state:\n{data}"


def _fallback_explain(state: ModelState, error: str) -> str:
    """Template fallback if Groq is unavailable."""
    sections = [f"*Groq API unavailable ({error}) — showing raw model summary:*\n"]

    if state.regime:
        sections.append(f"**Regime:** {state.regime} ({state.regime_confidence:.0%} confidence)")

    if state.backtest_stats:
        s = state.backtest_stats
        sections.append(
            f"**Backtest:** Sharpe {s.get('sharpe_strategy', 0):.2f} | "
            f"Alpha {s.get('ann_alpha', 0) * 100:+.1f}bp | "
            f"IR {s.get('information_ratio', 0):.2f} | "
            f"Max DD {s.get('max_drawdown_strategy', 0):.1%}"
        )

    if state.mc_var_95 is not None:
        sections.append(
            f"**Risk:** 95% VaR {state.mc_var_95 * 100:+.2f}% | "
            f"CVaR {(state.mc_cvar_95 or 0) * 100:+.2f}% | "
            f"P(Loss) {(state.mc_p_loss or 0):.1%}"
        )

    if state.posterior_returns:
        best = max(state.posterior_returns, key=state.posterior_returns.get)
        sections.append(
            f"**BL:** Highest posterior return: {best} "
            f"({state.posterior_returns[best] * 100:+.2f}%)"
        )

    return "\n\n".join(sections)
