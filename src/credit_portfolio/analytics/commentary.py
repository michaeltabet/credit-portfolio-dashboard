"""LLM commentary generator (Claude API + mock fallback)."""

import os

from credit_portfolio.config import load_config
from credit_portfolio.data.constants import LLM_MODEL_ID, LLM_MAX_TOKENS
from credit_portfolio.analytics.attribution import AttributionReport, format_for_llm


SYSTEM_PROMPT = """You are a senior portfolio analyst at a systematic index provider.
Your job is to write the quarterly rebalancing commentary for institutional clients.

VOICE AND REGISTER:
- Professional, clear, and precise — the register of the Financial Analysts Journal
- Write for a sophisticated investor (CIO of a pension fund, asset manager)
- Never mention "optimizer", "CVXPY", "KKT conditions", or any technical solver language
- Translate factor scores into investment language:
    * "z_dts" = "duration-times-spread exposure" or "spread duration risk"
    * "z_value" = "relative spread valuation" or "spread cheapness versus rating peers"
    * "z_momentum" = "spread momentum" or "trailing return trend"
- Mention specific sectors and the factor drivers behind moves
- If constraints became binding, explain in plain English what that means for the portfolio
- Do NOT say the strategy "decided" or "chose" — say "the rebalancing resulted in" or
  "the updated factor scores led to increased exposure to"

LENGTH: Exactly two paragraphs. First paragraph covers sector-level changes.
Second paragraph covers factor exposure shifts and any binding constraints.
Total: 150–220 words.

OUTPUT: Plain text only. No bullet points. No headers. No markdown."""


def generate_commentary(
    report    : AttributionReport,
    model     : str | None = None,
    max_tokens: int | None = None,
) -> str:
    """Generate quarterly commentary via Claude API."""
    import anthropic

    cfg = load_config()
    llm_cfg = cfg.get("llm", {})
    if model is None:
        model = llm_cfg.get("model", LLM_MODEL_ID)
    if max_tokens is None:
        max_tokens = llm_cfg.get("max_tokens", LLM_MAX_TOKENS)

    payload = format_for_llm(report)
    user_message = (
        f"Please write the quarterly rebalancing commentary for the "
        f"{report.rebal_date} rebalancing of our systematic multi-factor "
        f"investment-grade credit strategy. Use the attribution data below.\n\n"
        f"{payload}"
    )

    client = anthropic.Anthropic()
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return message.content[0].text


def generate_commentary_mock(report: AttributionReport) -> str:
    """Mock commentary generator for demonstration without API key."""
    sector_shifts = report.sector_shifts
    top_increase = sector_shifts[sector_shifts["change"] > 0].head(2)
    top_decrease = sector_shifts[sector_shifts["change"] < 0].tail(2)

    inc_sectors = " and ".join(top_increase["sector"].tolist()) if len(top_increase) > 0 else "selected sectors"
    dec_sectors = " and ".join(top_decrease["sector"].tolist()) if len(top_decrease) > 0 else "other sectors"

    dts_chg = report.factor_delta.get("z_dts", {}).get("change", 0)
    mom_chg = report.factor_delta.get("z_momentum", {}).get("change", 0)
    val_chg = report.factor_delta.get("z_value", {}).get("change", 0)

    dts_dir = "increased" if dts_chg > 0 else "decreased"
    mom_dir = "improved" if mom_chg > 0 else "deteriorated"

    top_add = report.top_adds.iloc[0] if len(report.top_adds) > 0 else None
    top_red = report.top_reduces.iloc[0] if len(report.top_reduces) > 0 else None

    constraint_text = ""
    if report.new_bindings:
        constraint_text = (
            f" The quality floor constraint, which was not binding in the prior period, "
            f"became active this quarter, resulting in the exclusion of "
            f"{report.n_bonds_t0 - report.n_bonds_t1} "
            f"names that no longer met the minimum issuer quality threshold."
        )

    para1 = (
        f"During the {report.rebal_date} rebalancing, the strategy increased its allocation "
        f"to {inc_sectors} while reducing exposure to {dec_sectors}. "
        f"This rotation reflects updated factor scores across the investment-grade universe: "
        f"{'duration-times-spread positioning and relative value signals were the primary drivers of the allocation shift, with the largest additions in names offering attractive spread compensation per unit of spread duration.' if top_add is not None and top_add['dominant_factor'] == 'DTS' else 'spread momentum and valuation signals were the primary drivers of the allocation shift, with the largest additions in names where trailing returns and relative spread cheapness signalled improving credit fundamentals.'}"
    )

    para2 = (
        f"At the factor level, the portfolio's duration-times-spread exposure {dts_dir} over the period, "
        f"while spread momentum signals {mom_dir} — reflecting "
        f"{'positive trailing returns in higher-spread segments consistent with spread compression' if mom_chg > 0 else 'spread widening in cyclical and commodity-linked sectors, reducing the attractiveness of momentum-driven positioning there'}. "
        f"Relative value signals {'remained supportive, with a subset of names continuing to trade wide of their rating-peer average on a log-spread basis' if val_chg >= 0 else 'compressed as spreads rallied broadly, reducing the dispersion of value opportunities across the universe'}."
        f"{constraint_text} "
        f"Overall factor tilts remain consistent with the strategy's objective of maintaining "
        f"balanced multi-factor exposure within the prescribed duration and sector neutrality bounds."
    )

    return para1 + "\n\n" + para2
