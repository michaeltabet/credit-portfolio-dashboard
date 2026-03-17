"""Rebalancing attribution between two periods."""

import numpy as np
import pandas as pd
from dataclasses import dataclass

from credit_portfolio.data.constants import (
    ATTRIBUTION_TOP_N, ATTRIBUTION_META_COLS, OPT_FACTOR_WEIGHTS,
)


@dataclass
class AttributionReport:
    top_adds     : pd.DataFrame
    top_reduces  : pd.DataFrame
    sector_shifts: pd.DataFrame
    factor_delta : dict
    binding_t0   : list
    binding_t1   : list
    new_bindings : list
    n_bonds_t0   : int
    n_bonds_t1   : int
    rebal_date   : str
    prior_date   : str


def attribute(
    df_t0, df_t1, result_t0, result_t1,
    rebal_date: str = "June 2025",
    prior_date: str = "March 2025",
    top_n: int = ATTRIBUTION_TOP_N,
) -> AttributionReport:
    """Produce full attribution between two rebalancing periods."""
    w0 = result_t0.weights.rename("w_t0")
    w1 = result_t1.weights.rename("w_t1")

    merged = pd.concat([w0, w1], axis=1).fillna(0.0)
    merged["delta"] = merged["w_t1"] - merged["w_t0"]

    meta_cols = list(ATTRIBUTION_META_COLS)

    def _get_meta(df):
        return df[meta_cols].set_index("bond_id")

    meta = _get_meta(df_t1).combine_first(_get_meta(df_t0))
    merged = merged.join(meta)

    adds = merged[merged["delta"] > 0].sort_values("delta", ascending=False)
    reduces = merged[merged["delta"] < 0].sort_values("delta")

    def _dominant_factor(row):
        scores = {
            "DTS"     : row.get("z_dts", 0),
            "value"   : row.get("z_value", 0),
            "momentum": row.get("z_momentum", 0),
        }
        return max(scores, key=scores.get)

    def _build_table(df_sub, n):
        rows = []
        for bond_id, row in df_sub.head(n).iterrows():
            rows.append({
                "bond_id"       : bond_id,
                "sector"        : row.get("sector", "Unknown"),
                "rating"        : row.get("rating", "N/A"),
                "weight_change" : round(row["delta"] * 100, 2),
                "dominant_factor": _dominant_factor(row),
                "z_composite"   : round(row.get("z_composite", 0), 2),
                "oas_bp"        : round(row.get("oas_bp", 0), 1),
                "spread_6m_chg" : round(row.get("spread_6m_chg", 0), 1),
            })
        return pd.DataFrame(rows)

    top_adds = _build_table(adds, top_n)
    top_reduces = _build_table(reduces, top_n)

    def _sector_weights(w_series, df):
        out = {}
        for s in df["sector"].unique():
            bonds = df[df["sector"] == s]["bond_id"].values
            out[s] = w_series.reindex(bonds).fillna(0).sum()
        return pd.Series(out)

    sec_t0 = _sector_weights(result_t0.weights, df_t0)
    sec_t1 = _sector_weights(result_t1.weights, df_t1)
    all_sectors = sec_t0.index.union(sec_t1.index)

    sector_shifts = pd.DataFrame({
        "sector"   : all_sectors,
        "weight_t0": sec_t0.reindex(all_sectors).fillna(0).round(4).values,
        "weight_t1": sec_t1.reindex(all_sectors).fillna(0).round(4).values,
    })
    sector_shifts["change"] = (sector_shifts["weight_t1"] - sector_shifts["weight_t0"]).round(4)
    sector_shifts = sector_shifts.sort_values("change", ascending=False)

    factor_delta = {}
    for f in list(OPT_FACTOR_WEIGHTS.keys()):
        e0 = result_t0.factor_exposures.get(f, 0)
        e1 = result_t1.factor_exposures.get(f, 0)
        factor_delta[f] = {
            "t0"    : round(e0, 4),
            "t1"    : round(e1, 4),
            "change": round(e1 - e0, 4),
        }

    new_bindings = [c for c in result_t1.binding_constraints
                    if c not in result_t0.binding_constraints]

    return AttributionReport(
        top_adds=top_adds,
        top_reduces=top_reduces,
        sector_shifts=sector_shifts,
        factor_delta=factor_delta,
        binding_t0=result_t0.binding_constraints,
        binding_t1=result_t1.binding_constraints,
        new_bindings=new_bindings,
        n_bonds_t0=len(result_t0.weights),
        n_bonds_t1=len(result_t1.weights),
        rebal_date=rebal_date,
        prior_date=prior_date,
    )


def format_for_llm(report: AttributionReport) -> str:
    """Serialise AttributionReport into structured plain-text for LLM prompt."""
    lines = []
    lines.append(f"REBALANCING DATE: {report.rebal_date}")
    lines.append(f"PRIOR DATE: {report.prior_date}")
    lines.append("")

    lines.append("=== SECTOR SHIFTS (weight change in %) ===")
    for _, row in report.sector_shifts.iterrows():
        direction = "increased" if row["change"] > 0 else "reduced"
        lines.append(
            f"  {row['sector']:20s}: {direction} by {abs(row['change']*100):.1f}pp "
            f"(from {row['weight_t0']*100:.1f}% to {row['weight_t1']*100:.1f}%)"
        )

    lines.append("")
    lines.append("=== TOP 5 ADDITIONS (by weight change) ===")
    for _, row in report.top_adds.iterrows():
        lines.append(
            f"  Bond {row['bond_id']} | Sector: {row['sector']} | Rating: {row['rating']} | "
            f"+{row['weight_change']:.2f}pp | Dominant factor: {row['dominant_factor']} | "
            f"Composite Z: {row['z_composite']:+.2f} | "
            f"OAS: {row['oas_bp']:.0f}bp | "
            f"6m spread change: {row['spread_6m_chg']:+.1f}bp"
        )

    lines.append("")
    lines.append("=== TOP 5 REDUCTIONS (by weight change) ===")
    for _, row in report.top_reduces.iterrows():
        lines.append(
            f"  Bond {row['bond_id']} | Sector: {row['sector']} | Rating: {row['rating']} | "
            f"{row['weight_change']:.2f}pp | Dominant factor: {row['dominant_factor']} | "
            f"Composite Z: {row['z_composite']:+.2f} | "
            f"OAS: {row['oas_bp']:.0f}bp | "
            f"6m spread change: {row['spread_6m_chg']:+.1f}bp"
        )

    lines.append("")
    lines.append("=== FACTOR EXPOSURE SHIFTS ===")
    factor_labels = {
        "z_dts"     : "DTS (duration × spread)",
        "z_value"   : "Value (excess log spread vs peers)",
        "z_momentum": "Momentum (trailing total return)",
    }
    for f, label in factor_labels.items():
        d = report.factor_delta.get(f, {})
        chg = d.get("change", 0)
        direction = "improved" if chg > 0 else "weakened"
        lines.append(
            f"  {label:35s}: {direction} ({d.get('t0', 0):+.4f} -> {d.get('t1', 0):+.4f})"
        )

    lines.append("")
    if report.new_bindings:
        lines.append("=== NEWLY BINDING CONSTRAINTS ===")
        for c in report.new_bindings:
            lines.append(f"  {c}")
    else:
        lines.append("=== NEWLY BINDING CONSTRAINTS: none ===")

    lines.append("")
    lines.append(f"Universe size: {report.n_bonds_t0} bonds (prior) -> {report.n_bonds_t1} bonds (current)")

    return "\n".join(lines)
