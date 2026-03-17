"""
Model Explainer Agent — generates plain-English explanations of model
outputs, portfolio decisions, and methodology.

This module provides two capabilities:
  1. explain_current_state() — summarize the full model state in plain text
  2. answer_question() — respond to natural-language questions about the model

Uses Claude API for natural language generation when available,
falls back to template-based explanations otherwise.
"""

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


def explain_current_state(state: ModelState) -> str:
    """Generate a full plain-English explanation of the current model state."""
    sections = []

    # ── Regime ─────────────────────────────────────────────────────
    if state.regime:
        regime_text = {
            "COMPRESSION": (
                "The HMM identifies the current regime as **Compression** — spreads are tight, "
                "volatility is low, and mean-reversion dynamics are strong. In this environment, "
                "the model trusts its equilibrium prior more (low τ) and treats Prophet forecasts "
                "as relatively reliable (low ω). This means the portfolio leans toward the "
                "market-implied allocation with modest factor tilts."
            ),
            "NORMAL": (
                "The HMM identifies the current regime as **Normal** — spreads are near their "
                "long-run average with moderate volatility. The model uses standard BL parameters "
                "(τ = 0.025, ω = 1.0), balancing equilibrium priors against forecast views equally. "
                "Factor signals have the most influence in this regime because neither the prior "
                "nor the views are dominant."
            ),
            "STRESS": (
                "The HMM identifies the current regime as **Stress** — spreads are wide, "
                "volatility is elevated, and tail risk is high. The model increases prior uncertainty "
                "(high τ) and dampens forecast confidence (high ω). This is a defensive posture: "
                "the portfolio should rotate toward higher-quality buckets and reduce active bets."
            ),
        }
        conf_pct = f"{state.regime_confidence:.0%}" if state.regime_confidence > 0 else "N/A"
        sections.append(f"## Market Regime\n\n{regime_text.get(state.regime, 'Unknown regime.')}\n\n"
                        f"Regime confidence: **{conf_pct}**")

    # ── BL Returns ─────────────────────────────────────────────────
    if state.posterior_returns:
        bl_lines = ["## Black-Litterman Expected Returns\n"]
        bl_lines.append("The BL model blends market equilibrium with our spread forecasts:\n")
        bl_lines.append("| Bucket | Prior | View | Posterior |")
        bl_lines.append("|--------|-------|------|-----------|")
        for bucket in ["AAA", "AA", "A", "BBB"]:
            prior = (state.prior_returns or {}).get(bucket, 0) * 100
            view = (state.view_returns or {}).get(bucket, 0) * 100
            post = state.posterior_returns.get(bucket, 0) * 100
            bl_lines.append(f"| {bucket} | {prior:+.2f}% | {view:+.2f}% | {post:+.2f}% |")

        # Interpretation
        posts = state.posterior_returns
        best_bucket = max(posts, key=posts.get)
        worst_bucket = min(posts, key=posts.get)
        bl_lines.append(f"\nThe model expects **{best_bucket}** to have the highest return and "
                        f"**{worst_bucket}** the lowest. The posterior tilts allocation accordingly.")
        sections.append("\n".join(bl_lines))

    # ── Prophet Forecasts ──────────────────────────────────────────
    if state.prophet_forecasts:
        pf_lines = ["## Prophet OAS Forecasts\n"]
        tightening = []
        widening = []
        for bucket, v in state.prophet_forecasts.items():
            delta_bp = v.get("delta_oas", 0) * 100
            if delta_bp < 0:
                tightening.append((bucket, delta_bp))
            else:
                widening.append((bucket, delta_bp))

        if tightening:
            names = ", ".join(f"**{b}** ({d:+.0f}bp)" for b, d in tightening)
            pf_lines.append(f"Prophet forecasts **spread tightening** for {names}. "
                            "Tightening generates positive price returns (bond prices rise).")
        if widening:
            names = ", ".join(f"**{b}** ({d:+.0f}bp)" for b, d in widening)
            pf_lines.append(f"\nProphet forecasts **spread widening** for {names}. "
                            "Widening generates negative price returns but increases future carry.")
        sections.append("\n".join(pf_lines))

    # ── Backtest ───────────────────────────────────────────────────
    if state.backtest_stats:
        s = state.backtest_stats
        bt_lines = ["## Historical Backtest\n"]
        sharpe = s.get("sharpe_strategy", 0)
        alpha = s.get("ann_alpha", 0) * 100
        ir = s.get("information_ratio", 0)
        hit = s.get("hit_rate", 0)

        if sharpe > 0.5:
            bt_lines.append(f"The strategy achieves a **{sharpe:.2f} Sharpe ratio** — "
                            "this is a reasonably strong risk-adjusted return for a constrained IG strategy.")
        elif sharpe > 0:
            bt_lines.append(f"The strategy achieves a **{sharpe:.2f} Sharpe ratio** — "
                            "positive but modest, typical for a 4-bucket rotation strategy.")
        else:
            bt_lines.append(f"The strategy has a **negative Sharpe ratio** ({sharpe:.2f}), "
                            "indicating the factor tilts are not adding value in this configuration.")

        bt_lines.append(f"\nAnnualized alpha: **{alpha:+.1f}bp** | "
                        f"Information ratio: **{ir:.2f}** | "
                        f"Hit rate: **{hit:.0%}**")

        if alpha > 0:
            bt_lines.append("\nThe positive alpha indicates the factor signals have "
                            "historically added value over the market-cap benchmark.")
        sections.append("\n".join(bt_lines))

    # ── Stress Test ────────────────────────────────────────────────
    if state.stress_price_impact:
        st_lines = [f"## Stress Test: {state.stress_scenario}\n"]
        worst_bucket = min(state.stress_price_impact, key=state.stress_price_impact.get)
        worst_impact = state.stress_price_impact[worst_bucket] * 100
        total_impact = sum(
            state.stress_price_impact[b] * (state.current_weights or {}).get(b, 0.25)
            for b in state.stress_price_impact
        ) * 100

        st_lines.append(f"Under the **{state.stress_scenario}** scenario:\n")
        st_lines.append(f"- Worst-hit bucket: **{worst_bucket}** ({worst_impact:+.1f}% price impact)")
        st_lines.append(f"- Estimated portfolio-level price impact: **{total_impact:+.1f}%**")

        if abs(total_impact) > 5:
            st_lines.append("\nThis is a severe scenario. Consider reducing active tilts "
                            "or increasing quality allocation.")
        sections.append("\n".join(st_lines))

    # ── Monte Carlo ────────────────────────────────────────────────
    if state.mc_var_95 is not None:
        mc_lines = ["## Monte Carlo Risk\n"]
        mc_lines.append(f"Based on Monte Carlo simulation of real historical returns:\n")
        mc_lines.append(f"- **95% VaR**: {state.mc_var_95 * 100:+.2f}% "
                        "(there's a 5% chance of losing more than this)")
        if state.mc_cvar_95 is not None:
            mc_lines.append(f"- **95% CVaR**: {state.mc_cvar_95 * 100:+.2f}% "
                            "(average loss in the worst 5% of scenarios)")
        if state.mc_p_loss is not None:
            mc_lines.append(f"- **P(Loss)**: {state.mc_p_loss:.1%} "
                            "(probability of any negative return over the horizon)")
        if state.mc_median_return is not None:
            mc_lines.append(f"- **Median return**: {state.mc_median_return * 100:+.2f}%")
        sections.append("\n".join(mc_lines))

    if not sections:
        return "No model state available. Run the analysis first."

    return "\n\n---\n\n".join(sections)


def answer_question(question: str, state: ModelState) -> str:
    """
    Answer a natural-language question about the model.

    This is a rule-based responder for common questions. For complex questions,
    consider routing to Claude API via the commentary module.
    """
    q = question.lower().strip()

    # Regime questions
    if any(w in q for w in ["regime", "hmm", "state", "market condition"]):
        if state.regime:
            explanation = {
                "COMPRESSION": "tight spreads and low volatility — the model is in a risk-on posture",
                "NORMAL": "average spreads and moderate volatility — the model uses balanced parameters",
                "STRESS": "wide spreads and high volatility — the model is defensive",
            }
            return (f"The current regime is **{state.regime}** ({state.regime_confidence:.0%} confidence). "
                    f"This means {explanation.get(state.regime, 'unknown conditions')}. "
                    f"The regime determines two key parameters: τ (how much we trust the equilibrium prior) "
                    f"and ω (how much we trust Prophet forecasts).")
        return "No regime data available. Run the HMM analysis first."

    # BL questions
    if any(w in q for w in ["black-litterman", "bl ", "expected return", "posterior", "prior"]):
        if state.posterior_returns:
            best = max(state.posterior_returns, key=state.posterior_returns.get)
            return (f"The Black-Litterman model combines the market equilibrium (prior) with Prophet "
                    f"forecast views to produce posterior expected returns. Currently, **{best}** has "
                    f"the highest posterior return ({state.posterior_returns[best]*100:+.2f}%). "
                    f"The posterior is a weighted average — in stress regimes it stays closer to the "
                    f"prior; in compression it incorporates more of the forecast view.")
        return "No BL data available. Run the analysis first."

    # Prophet questions
    if any(w in q for w in ["prophet", "forecast", "predict", "oas forecast"]):
        if state.prophet_forecasts:
            lines = ["Prophet forecasts OAS for each rating bucket using logistic-growth decomposition:\n"]
            for bucket, v in state.prophet_forecasts.items():
                delta = v.get("delta_oas", 0) * 100
                direction = "tightening" if delta < 0 else "widening"
                lines.append(f"- **{bucket}**: {v.get('current_oas', 0):.2f}% → "
                             f"{v.get('forecast_oas', 0):.2f}% ({delta:+.0f}bp {direction})")
            lines.append("\nTightening = positive for bond prices. Widening = negative but increases carry.")
            return "\n".join(lines)
        return "No Prophet data available. Run the analysis first."

    # Factor questions
    if any(w in q for w in ["factor", "dts", "value", "momentum", "signal", "composite"]):
        return (
            "The strategy uses three credit factors:\n\n"
            "1. **DTS (Duration × Spread)** — weight: 50% — measures spread-duration risk. "
            "Higher DTS = more compensation for spread risk.\n"
            "2. **Value** — weight: 25% — cross-sectional z-score of log(OAS). "
            "Wider spreads relative to peers = cheaper = expected to tighten.\n"
            "3. **Momentum** — weight: 25% — trailing 6-month cumulative excess return. "
            "Buckets with positive momentum tend to continue outperforming.\n\n"
            "These are combined into a composite z-score that tilts portfolio weights "
            "away from the market-cap benchmark."
        )

    # Risk questions
    if any(w in q for w in ["risk", "var", "cvar", "monte carlo", "drawdown", "loss"]):
        if state.mc_var_95 is not None:
            return (f"Based on Monte Carlo simulation:\n"
                    f"- 95% VaR: {state.mc_var_95*100:+.2f}% (5% chance of worse)\n"
                    f"- 95% CVaR: {(state.mc_cvar_95 or 0)*100:+.2f}% (average in worst 5%)\n"
                    f"- P(Loss): {(state.mc_p_loss or 0):.1%}\n\n"
                    f"VaR tells you the threshold; CVaR tells you how bad it gets past that threshold.")
        if state.backtest_stats:
            dd = state.backtest_stats.get("max_drawdown_strategy", 0)
            return f"Historical max drawdown: {dd:.1%}. Run Monte Carlo for forward-looking risk estimates."
        return "No risk data available. Run Monte Carlo analysis first."

    # Stress test questions
    if any(w in q for w in ["stress", "shock", "crisis", "scenario"]):
        if state.stress_price_impact:
            impacts = ", ".join(
                f"{b}: {v*100:+.1f}%" for b, v in state.stress_price_impact.items()
            )
            return (f"Under the **{state.stress_scenario}** scenario, the price impacts are:\n"
                    f"{impacts}\n\n"
                    f"These represent the immediate mark-to-market loss from the spread shock, "
                    f"calculated as -Duration × ΔOAS for each bucket.")
        return "No stress test data available. Run the stress test first."

    # Weight / allocation questions
    if any(w in q for w in ["weight", "allocation", "position", "portfolio"]):
        if state.current_weights:
            lines = ["Current portfolio allocation vs benchmark:\n"]
            for b in ["AAA", "AA", "A", "BBB"]:
                w = state.current_weights.get(b, 0)
                bw = (state.benchmark_weights or {}).get(b, 0)
                tilt = w - bw
                lines.append(f"- **{b}**: {w:.1%} (benchmark: {bw:.0%}, tilt: {tilt:+.1%})")
            return "\n".join(lines)
        return "No weight data available. Run the backtest first."

    # Generic / catch-all
    return (
        "I can explain the following aspects of the model:\n\n"
        "- **Regime** — What regime is the market in? How does it affect the model?\n"
        "- **Black-Litterman** — How are expected returns computed?\n"
        "- **Prophet** — What are the OAS forecasts?\n"
        "- **Factors** — How do DTS, Value, and Momentum work?\n"
        "- **Risk** — What are VaR/CVaR and the loss probability?\n"
        "- **Stress** — What happens under different shock scenarios?\n"
        "- **Weights** — What is the current allocation?\n\n"
        "Ask about any of these topics and I'll explain in detail."
    )
