"""Interactive Streamlit dashboard — 5-tab analytical dashboard.

All data is real FRED OAS / total return data. No synthetic bonds.

Tabs:
  1. HMM / Black-Litterman  — regime detection + BL posterior on real FRED data
  2. Historical Backtest     — real FRED OAS bucket-rotation backtest
  3. Prophet Forecasts       — per-bucket OAS forecasts with direction/uncertainty
  4. Stress Testing          — predefined + custom shock scenarios (bucket level)
  5. Monte Carlo             — bootstrap/parametric simulation with VaR/CVaR
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

from credit_portfolio.data.constants import (
    OPT_FACTOR_WEIGHTS,
    COLORS,
    # HMM
    HMM_N_STATES, HMM_MAX_ITER,
    # Prophet
    PROPHET_HORIZON_MONTHS, PROPHET_CHANGEPOINT_PRIOR_SCALE,
    PROPHET_SEASONALITY_PRIOR_SCALE, PROPHET_INTERVAL_WIDTH,
    # BL
    LAMBDA_RISK_AVERSION, PIPELINE_TAU, WITHIN_BUCKET_CORR,
    # Pipeline
    IG_ASSETS, ASSET_LABELS, IG_MARKET_WEIGHTS_ARRAY,
    DURATIONS, COV_WINDOW, BL_RIDGE_PENALTY, OAS_PCT_TO_BP, MONTHS_PER_YEAR,
    SECTORS, RATINGS,
)

# ── Colors ────────────────────────────────────────────────────────
BLUE = COLORS["primary"]
AMBER = COLORS["amber"]
GRAY = COLORS["neutral"]
GREEN = COLORS["green"]
RED = COLORS["accent"]

st.set_page_config(page_title="Credit Portfolio Dashboard", layout="wide")
st.title("Credit Portfolio Dashboard")

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("Configuration")

    # ── HMM Regime ────────────────────────────────────────────────
    with st.expander("HMM Regime Detection", expanded=True):
        hmm_n_states = st.slider("Hidden States", 2, 5, HMM_N_STATES)
        hmm_max_iter = st.slider("Max EM Iterations", 50, 500, HMM_MAX_ITER, 50)

    # ── Black-Litterman ───────────────────────────────────────────
    with st.expander("Black-Litterman"):
        bl_risk_aversion = st.slider(
            "Risk Aversion (lambda)", 0.5, 5.0, LAMBDA_RISK_AVERSION, 0.1,
        )
        bl_tau = st.slider("Tau (prior uncertainty)", 0.001, 0.10, PIPELINE_TAU, 0.001, format="%.3f")

    # ── Prophet Views ─────────────────────────────────────────────
    with st.expander("Prophet Forecasts"):
        prophet_horizon = st.slider("Forecast Horizon (months)", 1, 12, PROPHET_HORIZON_MONTHS)
        prophet_changepoint = st.slider(
            "Changepoint Prior Scale", 0.01, 0.50,
            PROPHET_CHANGEPOINT_PRIOR_SCALE, 0.01,
        )
        prophet_seasonality = st.slider(
            "Seasonality Prior Scale", 1.0, 15.0,
            PROPHET_SEASONALITY_PRIOR_SCALE, 0.5,
        )
        prophet_interval = st.slider(
            "Interval Width", 0.50, 0.99,
            PROPHET_INTERVAL_WIDTH, 0.05,
        )

    # ── Historical Backtest ───────────────────────────────────────
    with st.expander("Historical Backtest"):
        hist_tilt_strength = st.slider("Tilt Strength", 0.01, 0.30, 0.10, 0.01)
        hist_momentum_window = st.slider("Momentum Window (months)", 3, 12, 6)
        hist_tc_bps = st.slider("Transaction Cost (bps)", 0.0, 20.0, 5.0, 0.5)

    # ── Stress Testing ────────────────────────────────────────────
    with st.expander("Stress Testing"):
        stress_scenario = st.selectbox(
            "Scenario",
            ["Spread Widening +200bp", "BBB Crisis", "Fed Hike Shock", "COVID Replay", "Custom"],
        )
        if stress_scenario == "Custom":
            st.caption("Custom OAS shocks (bp) — will be converted to % for FRED data")
            custom_shocks_bp = {}
            for r in RATINGS:
                val = st.number_input(f"{r} shock (bp)", value=0, step=10, key=f"stress_{r}")
                if val != 0:
                    custom_shocks_bp[r] = val / 100.0  # convert bp to pct for FRED
        else:
            custom_shocks_bp = {}

    # ── Monte Carlo ───────────────────────────────────────────────
    with st.expander("Monte Carlo"):
        mc_n_sims = st.slider("Simulations", 100, 10000, 1000, 100)
        mc_horizon = st.slider("Horizon (months)", 3, 60, 12)
        mc_method = st.radio("Method", ["bootstrap", "parametric"], horizontal=True)
        mc_conf_levels = st.multiselect(
            "Confidence Levels",
            [90.0, 95.0, 99.0],
            default=[90.0, 95.0, 99.0],
        )

    st.divider()
    run_btn = st.button("Run Analysis", type="primary", use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════
def _layout(title: str, yaxis_title: str = "", **kwargs):
    return dict(
        title=dict(text=title, x=0, font=dict(size=14)),
        template="plotly_white",
        yaxis_title=yaxis_title,
        hovermode="x unified",
        margin=dict(l=60, r=20, t=40, b=40),
        legend=dict(orientation="h", y=-0.15),
        **kwargs,
    )


def _load_fred():
    """Load real FRED data (cached at session level)."""
    from credit_portfolio.config import load_config, resolve_csv_path
    from credit_portfolio.data.loader import load
    cfg = load_config()
    csv_path = resolve_csv_path(cfg)
    return load(str(csv_path))


# ══════════════════════════════════════════════════════════════════
# CACHED RUNNERS — all use real FRED data only
# ══════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _run_hmm_bl(hmm_n_states, hmm_max_iter, bl_risk_aversion, bl_tau):
    """Run HMM regime detection + Black-Litterman on real FRED data."""
    from credit_portfolio.models.hmm_regime import fit_hmm, get_current_regime, regime_summary

    df = _load_fred()

    # HMM on real FRED OAS data
    hmm_result = fit_hmm(df, n_states=hmm_n_states, n_iter=hmm_max_iter)
    regime_info = get_current_regime(hmm_result)

    # BL on real FRED data
    n = len(IG_ASSETS)
    ret_data = {}
    for a in IG_ASSETS:
        if a in df.columns:
            dur = DURATIONS.get(a, 7.0)
            ret_data[a] = -dur * df[a].diff(1) + df[a] / MONTHS_PER_YEAR
    Sigma = pd.DataFrame(ret_data).dropna().tail(COV_WINDOW).cov().values

    Pi = bl_risk_aversion * Sigma @ IG_MARKET_WEIGHTS_ARRAY

    Q_vec = np.zeros(n)
    for i, a in enumerate(IG_ASSETS):
        if a in df.columns:
            chg_3m = float(df[a].diff(3).iloc[-1])
            dur = DURATIONS.get(a, 7.0)
            Q_vec[i] = -dur * chg_3m / OAS_PCT_TO_BP

    P_mat = np.eye(n)
    omega_scale = hmm_result.omega_scale
    tS = bl_tau * Sigma
    Omega = np.diag([
        omega_scale * bl_tau * float(P_mat[i] @ tS @ P_mat[i])
        for i in range(n)
    ]) + np.eye(n) * BL_RIDGE_PENALTY

    tS_inv = np.linalg.inv(tS + np.eye(n) * BL_RIDGE_PENALTY)
    Om_inv = np.linalg.inv(Omega)
    A = tS_inv + P_mat.T @ Om_inv @ P_mat
    b = tS_inv @ Pi + P_mat.T @ Om_inv @ Q_vec

    try:
        mu_BL = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        mu_BL = Pi.copy()

    return {
        "hmm": hmm_result,
        "regime_info": regime_info,
        "regime_stats": regime_summary(hmm_result),
        "bl": {"Pi": Pi, "mu_BL": mu_BL, "Q_vec": Q_vec, "Sigma": Sigma},
        "monthly": df,
    }


@st.cache_data(show_spinner=False)
def _run_hist_backtest(tilt_strength, momentum_window, tc_bps):
    """Run bucket-rotation backtest on real FRED data."""
    from credit_portfolio.backtests.bucket_backtest import run_backtest, BacktestConfig

    df = _load_fred()
    bt_cfg = BacktestConfig(
        tilt_strength=tilt_strength,
        momentum_window=momentum_window,
        tc_bps=tc_bps,
    )
    result = run_backtest(df, config=bt_cfg)
    return result


@st.cache_data(show_spinner=False)
def _run_prophet(prophet_horizon, prophet_changepoint, prophet_seasonality, prophet_interval):
    """Run Prophet forecasts on real FRED data."""
    from credit_portfolio.models.prophet_views import generate_all_views

    df = _load_fred()
    views = generate_all_views(
        df,
        horizon_months=prophet_horizon,
        changepoint_prior_scale=prophet_changepoint,
        seasonality_prior_scale=prophet_seasonality,
        interval_width=prophet_interval,
    )
    return views


@st.cache_data(show_spinner=False)
def _run_stress(scenario, custom_shocks_tuple, tilt_strength, momentum_window):
    """Run bucket-level stress test on real FRED data."""
    from credit_portfolio.analytics.stress_test import run_stress_test
    from credit_portfolio.backtests.bucket_backtest import BacktestConfig

    df = _load_fred()
    config = BacktestConfig(tilt_strength=tilt_strength, momentum_window=momentum_window)

    custom_shocks = dict(custom_shocks_tuple) if custom_shocks_tuple else None

    return run_stress_test(
        monthly=df,
        scenario=scenario,
        config=config,
        custom_shocks=custom_shocks if scenario == "Custom" else None,
    )


@st.cache_data(show_spinner=False)
def _run_monte_carlo(tilt_strength, momentum_window, tc_bps,
                     mc_n_sims, mc_horizon, mc_method, mc_conf_levels_tuple):
    """Run Monte Carlo on real FRED backtest returns."""
    from credit_portfolio.backtests.bucket_backtest import run_backtest, BacktestConfig
    from credit_portfolio.analytics.monte_carlo import run_monte_carlo

    df = _load_fred()
    bt_cfg = BacktestConfig(tilt_strength=tilt_strength, momentum_window=momentum_window, tc_bps=tc_bps)
    bt_result = run_backtest(df, config=bt_cfg)

    return run_monte_carlo(
        historical_returns=bt_result.monthly_returns_strategy,
        n_sims=mc_n_sims,
        horizon_months=mc_horizon,
        method=mc_method,
        confidence_levels=list(mc_conf_levels_tuple),
    )


# ══════════════════════════════════════════════════════════════════
# MAIN AREA — 5 TABS (all real FRED data)
# ══════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "HMM / Black-Litterman",
    "Historical Backtest",
    "Prophet Forecasts",
    "Stress Testing",
    "Monte Carlo",
    "Ask the Model",
])


# ══════════════════════════════════════════════════════════════════
# TAB 1: HMM / BLACK-LITTERMAN (real FRED data)
# ══════════════════════════════════════════════════════════════════
with tab1:
    if run_btn or "hmm_bl" not in st.session_state:
        with st.spinner("Fitting HMM + Black-Litterman on FRED data..."):
            try:
                hmm_bl = _run_hmm_bl(hmm_n_states, hmm_max_iter, bl_risk_aversion, bl_tau)
                st.session_state["hmm_bl"] = hmm_bl
            except Exception as e:
                st.error(f"HMM/BL failed: {e}")
                st.session_state["hmm_bl"] = None

    hmm_bl = st.session_state.get("hmm_bl")
    if hmm_bl is None:
        st.info("Click **Run Analysis** to fit HMM + BL on real FRED data.")
    else:
        hmm_result = hmm_bl["hmm"]
        bl = hmm_bl["bl"]

        # KPI row
        p1, p2, p3 = st.columns(3)
        p1.metric("Current Regime", hmm_bl["regime_info"]["regime"])
        _ri = hmm_bl["regime_info"]
        _regime_prob = {
            "COMPRESSION": _ri.get("p_compression", 0),
            "NORMAL": _ri.get("p_normal", 0),
            "STRESS": _ri.get("p_stress", 0),
        }.get(_ri["regime"], 0)
        p2.metric("Regime Confidence", f"{_regime_prob:.1%}")
        p3.metric("Omega Scale", f"{hmm_result.omega_scale:.1f}")

        hb1, hb2, hb3 = st.tabs(["HMM Regimes", "Black-Litterman", "Transition Matrix"])

        # ── HMM Regimes ──────────────────────────────────────────
        with hb1:
            st.subheader("Regime Statistics")
            st.dataframe(hmm_bl["regime_stats"], use_container_width=True)

            probs = hmm_result.state_probs
            fig_hmm = go.Figure()
            regime_colors = {0: BLUE, 1: GRAY, 2: RED}
            regime_names = {0: "Compression", 1: "Normal", 2: "Stress"}
            for state in range(min(hmm_n_states, 3)):
                if state in probs.columns:
                    fig_hmm.add_trace(go.Scatter(
                        x=probs.index, y=probs[state].values,
                        name=regime_names.get(state, f"State {state}"),
                        stackgroup="one",
                        line=dict(width=0.5, color=regime_colors.get(state, GRAY)),
                    ))
            fig_hmm.update_layout(**_layout("HMM Regime Probabilities (FRED IG OAS)", "Probability"))
            fig_hmm.update_yaxes(range=[0, 1])
            st.plotly_chart(fig_hmm, use_container_width=True)

        # ── Black-Litterman ──────────────────────────────────────
        with hb2:
            Pi = bl["Pi"]
            mu_BL = bl["mu_BL"]
            Q_vec = bl["Q_vec"]

            fig_bl = go.Figure()
            labels = [ASSET_LABELS.get(a, a) for a in IG_ASSETS]

            fig_bl.add_trace(go.Bar(x=labels, y=Pi * 100, name="Prior (Pi)", marker_color=GRAY))
            fig_bl.add_trace(go.Bar(x=labels, y=Q_vec * 100, name="Views (Q)", marker_color=AMBER))
            fig_bl.add_trace(go.Bar(x=labels, y=mu_BL * 100, name="Posterior (BL)", marker_color=BLUE))
            fig_bl.update_layout(
                barmode="group",
                **_layout("Black-Litterman: Prior vs Views vs Posterior (FRED Data)", "Expected Return (%)"),
            )
            st.plotly_chart(fig_bl, use_container_width=True)

            st.caption(f"Regime: **{hmm_bl['regime_info']['regime']}** | "
                       f"Tau: {bl_tau:.3f} | Risk Aversion: {bl_risk_aversion:.1f}")

        # ── Transition Matrix ─────────────────────────────────────
        with hb3:
            labels = [regime_names.get(i, f"State {i}") for i in range(hmm_result.transition_matrix.shape[0])]
            tm_df = pd.DataFrame(
                hmm_result.transition_matrix,
                index=labels, columns=labels,
            ).round(3)
            st.dataframe(tm_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# TAB 2: HISTORICAL BACKTEST (real FRED data)
# ══════════════════════════════════════════════════════════════════
with tab2:
    if run_btn or "hist_result" not in st.session_state:
        with st.spinner("Running historical backtest on FRED data..."):
            try:
                hist_result = _run_hist_backtest(hist_tilt_strength, hist_momentum_window, hist_tc_bps)
                st.session_state["hist_result"] = hist_result
            except Exception as e:
                st.error(f"Historical backtest failed: {e}")
                st.session_state["hist_result"] = None

    hist_result = st.session_state.get("hist_result")
    if hist_result is None:
        st.info("Click **Run Analysis** to run historical backtest on real FRED data.")
    else:
        stats = hist_result.stats

        # KPI row
        h1, h2, h3, h4, h5 = st.columns(5)
        h1.metric("Strategy Sharpe", f"{stats['sharpe_strategy']:.2f}")
        h2.metric("Benchmark Sharpe", f"{stats['sharpe_benchmark']:.2f}")
        h3.metric("Alpha (ann.)", f"{stats['ann_alpha']:+.2%}")
        h4.metric("Max DD", f"{stats['max_drawdown_strategy']:.1%}")
        h5.metric("Info Ratio", f"{stats['information_ratio']:.2f}")

        ht1, ht2, ht3, ht4 = st.tabs(["Cumulative Returns", "Factor Signals", "Weight Allocation", "Stats"])

        # ── Cumulative Returns ────────────────────────────────────
        with ht1:
            fig_hc = go.Figure()
            fig_hc.add_trace(go.Scatter(
                x=hist_result.cumulative_strategy.index,
                y=hist_result.cumulative_strategy.values,
                name="Strategy", line=dict(color=BLUE, width=2),
            ))
            fig_hc.add_trace(go.Scatter(
                x=hist_result.cumulative_benchmark.index,
                y=hist_result.cumulative_benchmark.values,
                name="Benchmark (Mkt-Cap)", line=dict(color=GRAY, width=2, dash="dash"),
            ))
            fig_hc.update_layout(**_layout("Cumulative Returns — FRED OAS Data", "Growth of $1"))
            st.plotly_chart(fig_hc, use_container_width=True)

        # ── Factor Signals ────────────────────────────────────────
        with ht2:
            sig_df = hist_result.signals_history
            for factor_prefix in ["z_dts", "z_value", "z_momentum", "composite"]:
                cols = [c for c in sig_df.columns if c.startswith(factor_prefix)]
                if cols:
                    fig_sig = go.Figure()
                    colors_list = [BLUE, AMBER, GREEN, RED]
                    for i, col in enumerate(cols):
                        bucket_name = col.replace(f"{factor_prefix}_", "")
                        fig_sig.add_trace(go.Scatter(
                            x=sig_df.index, y=sig_df[col].values,
                            name=bucket_name,
                            line=dict(color=colors_list[i % len(colors_list)], width=1.5),
                        ))
                    fig_sig.add_hline(y=0, line_dash="solid", line_color="black", line_width=0.5)
                    fig_sig.update_layout(**_layout(
                        f"{factor_prefix.replace('z_', '').title()} Signal by Rating Bucket", "Z-Score"))
                    st.plotly_chart(fig_sig, use_container_width=True)

        # ── Weight Allocation ─────────────────────────────────────
        with ht3:
            fig_wa = go.Figure()
            colors_stack = [BLUE, AMBER, GREEN, RED]
            for i, col in enumerate(hist_result.weights_history.columns):
                fig_wa.add_trace(go.Scatter(
                    x=hist_result.weights_history.index,
                    y=hist_result.weights_history[col].values,
                    name=col, stackgroup="one",
                    line=dict(width=0.5, color=colors_stack[i % len(colors_stack)]),
                ))
            fig_wa.update_layout(**_layout("Rating Bucket Allocation Over Time", "Weight"))
            fig_wa.update_yaxes(range=[0, 1])
            st.plotly_chart(fig_wa, use_container_width=True)

        # ── Stats ─────────────────────────────────────────────────
        with ht4:
            stat_rows = [{
                "Metric": k.replace("_", " ").title(),
                "Value": f"{v:.4f}" if isinstance(v, float) else str(v),
            } for k, v in stats.items()]
            st.dataframe(pd.DataFrame(stat_rows).set_index("Metric"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# TAB 3: PROPHET FORECASTS (real FRED data)
# ══════════════════════════════════════════════════════════════════
with tab3:
    if run_btn or "prophet_result" not in st.session_state:
        with st.spinner("Running Prophet forecasts on FRED data (~30s)..."):
            try:
                views = _run_prophet(
                    prophet_horizon, prophet_changepoint,
                    prophet_seasonality, prophet_interval,
                )
                st.session_state["prophet_result"] = views
            except Exception as e:
                st.error(f"Prophet forecasts failed: {e}")
                st.session_state["prophet_result"] = None

    views = st.session_state.get("prophet_result")
    if views is None or len(views) == 0:
        st.info("Click **Run Analysis** to generate Prophet forecasts on real FRED data.")
    else:
        # Summary table
        table_rows = []
        for bucket, v in views.items():
            delta_bp = v["delta_oas"] * 100
            table_rows.append({
                "Bucket": bucket,
                "Current OAS (%)": f"{v['current_oas']:.2f}",
                "Forecast OAS (%)": f"{v['forecast_oas']:.2f}",
                "Delta (bp)": f"{delta_bp:+.1f}",
                "Uncertainty (%)": f"{v['uncertainty']:.2f}",
                "Expected Return (%)": f"{v['expected_return'] * 100:+.2f}",
            })
        st.subheader("Prophet Forecast Summary (FRED Data)")
        st.dataframe(pd.DataFrame(table_rows).set_index("Bucket"), use_container_width=True)

        pt1, pt2 = st.tabs(["Current vs Forecast OAS", "Direction Chart"])

        # ── Bar chart: current vs forecast ────────────────────────
        with pt1:
            buckets = list(views.keys())
            current_oas = [views[b]["current_oas"] for b in buckets]
            forecast_oas = [views[b]["forecast_oas"] for b in buckets]

            fig_pbar = go.Figure()
            fig_pbar.add_trace(go.Bar(x=buckets, y=current_oas, name="Current OAS", marker_color=GRAY))
            fig_pbar.add_trace(go.Bar(x=buckets, y=forecast_oas, name="Forecast OAS", marker_color=BLUE))
            fig_pbar.update_layout(barmode="group", **_layout("Current vs Forecast OAS (FRED)", "OAS (%)"))
            st.plotly_chart(fig_pbar, use_container_width=True)

        # ── Direction chart ───────────────────────────────────────
        with pt2:
            buckets = list(views.keys())
            deltas = [views[b]["delta_oas"] * 100 for b in buckets]
            colors = [GREEN if d < 0 else RED for d in deltas]

            fig_dir = go.Figure()
            fig_dir.add_trace(go.Bar(
                y=buckets, x=deltas, orientation="h",
                marker_color=colors,
                text=[f"{d:+.1f}bp" for d in deltas],
                textposition="outside",
            ))
            fig_dir.update_layout(**_layout(
                "Forecast Direction (Tightening = Green, Widening = Red)", ""))
            fig_dir.update_xaxes(title_text="Delta OAS (bp)")
            st.plotly_chart(fig_dir, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# TAB 4: STRESS TESTING (real FRED data, bucket level)
# ══════════════════════════════════════════════════════════════════
with tab4:
    if run_btn or "stress_result" not in st.session_state:
        with st.spinner("Running stress test on FRED data..."):
            try:
                stress_result = _run_stress(
                    stress_scenario,
                    tuple(sorted(custom_shocks_bp.items())) if custom_shocks_bp else (),
                    hist_tilt_strength,
                    hist_momentum_window,
                )
                st.session_state["stress_result"] = stress_result
            except Exception as e:
                st.error(f"Stress test failed: {e}")
                st.session_state["stress_result"] = None

    stress_result = st.session_state.get("stress_result")
    if stress_result is None:
        st.info("Click **Run Analysis** to run stress test on real FRED data.")
    else:
        st.subheader(f"Scenario: {stress_result.scenario_name}")
        st.caption(stress_result.description)

        st1, st2, st3 = st.tabs(["Bucket Weights", "OAS Impact", "Return Impact"])

        # ── Bucket weights: baseline vs stressed ──────────────────
        with st1:
            wc = stress_result.weight_changes
            fig_bw = go.Figure()
            fig_bw.add_trace(go.Bar(x=wc["Bucket"], y=wc["Market Weight"], name="Market (Benchmark)", marker_color=GRAY))
            fig_bw.add_trace(go.Bar(x=wc["Bucket"], y=wc["Baseline Weight"], name="Baseline (Signal-Tilted)", marker_color=BLUE))
            fig_bw.add_trace(go.Bar(x=wc["Bucket"], y=wc["Stressed Weight"], name="Stressed", marker_color=RED))
            fig_bw.update_layout(barmode="group", **_layout("Rating Bucket Weights: Baseline vs Stressed", "Weight"))
            st.plotly_chart(fig_bw, use_container_width=True)

            # Weight change table
            st.subheader("Weight Changes")
            display_wc = wc.copy()
            for col in ["Market Weight", "Baseline Weight", "Stressed Weight", "Weight Change"]:
                display_wc[col] = display_wc[col].apply(lambda x: f"{x:.4f}")
            for col in ["Baseline OAS (%)", "Stressed OAS (%)"]:
                display_wc[col] = display_wc[col].apply(lambda x: f"{x:.2f}" if isinstance(x, float) else x)
            st.dataframe(display_wc.set_index("Bucket"), use_container_width=True)

        # ── OAS impact ────────────────────────────────────────────
        with st2:
            buckets = list(stress_result.baseline_oas.keys())
            base_oas = [stress_result.baseline_oas[b] for b in buckets]
            stress_oas = [stress_result.stressed_oas[b] for b in buckets]

            fig_oas = go.Figure()
            fig_oas.add_trace(go.Bar(x=buckets, y=base_oas, name="Baseline OAS", marker_color=BLUE))
            fig_oas.add_trace(go.Bar(x=buckets, y=stress_oas, name="Stressed OAS", marker_color=RED))
            fig_oas.update_layout(barmode="group", **_layout("OAS Levels: Baseline vs Stressed (FRED)", "OAS (%)"))
            st.plotly_chart(fig_oas, use_container_width=True)

        # ── Return impact ─────────────────────────────────────────
        with st3:
            buckets = list(stress_result.price_impact.keys())
            impact_rows = []
            for b in buckets:
                impact_rows.append({
                    "Bucket": b,
                    "Shock (bp)": f"{stress_result.shocks_applied.get(b, 0) * 100:+.0f}",
                    "Price Impact": f"{stress_result.price_impact[b] * 100:+.2f}%",
                    "Baseline Carry (mo)": f"{stress_result.baseline_carry[b] * 100:.3f}%",
                    "Stressed Carry (mo)": f"{stress_result.stressed_carry[b] * 100:.3f}%",
                })
            st.dataframe(pd.DataFrame(impact_rows).set_index("Bucket"), use_container_width=True)

            # Price impact bar chart
            impacts = [stress_result.price_impact[b] * 100 for b in buckets]
            fig_imp = go.Figure()
            fig_imp.add_trace(go.Bar(
                x=buckets, y=impacts,
                marker_color=[RED if v < 0 else GREEN for v in impacts],
                text=[f"{v:+.2f}%" for v in impacts],
                textposition="outside",
            ))
            fig_imp.update_layout(**_layout("Price Impact by Rating Bucket", "Price Return (%)"))
            st.plotly_chart(fig_imp, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# TAB 5: MONTE CARLO (uses real FRED backtest returns)
# ══════════════════════════════════════════════════════════════════
with tab5:
    if run_btn or "mc_result" not in st.session_state:
        with st.spinner("Running Monte Carlo simulation..."):
            try:
                mc_result = _run_monte_carlo(
                    hist_tilt_strength, hist_momentum_window, hist_tc_bps,
                    mc_n_sims, mc_horizon, mc_method,
                    tuple(mc_conf_levels),
                )
                st.session_state["mc_result"] = mc_result
            except Exception as e:
                st.error(f"Monte Carlo failed: {e}")
                st.session_state["mc_result"] = None

    mc_result = st.session_state.get("mc_result")
    if mc_result is None:
        st.info("Click **Run Analysis** to run Monte Carlo on real FRED backtest returns.")
    else:
        st.subheader(f"Monte Carlo ({mc_result.method.title()}) — {mc_result.n_sims:,} sims, {mc_result.horizon_months}mo horizon")

        mc1, mc2, mc3 = st.tabs(["Fan Chart", "Return Distribution", "Risk Metrics"])

        months = np.arange(1, mc_result.horizon_months + 1)

        # ── Fan chart ─────────────────────────────────────────────
        with mc1:
            fig_fan = go.Figure()

            # 5-95 band
            fig_fan.add_trace(go.Scatter(
                x=months, y=mc_result.percentile_bands[95] * 100,
                mode="lines", line=dict(width=0), showlegend=False,
            ))
            fig_fan.add_trace(go.Scatter(
                x=months, y=mc_result.percentile_bands[5] * 100,
                mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor="rgba(27,79,130,0.1)",
                name="5-95th pctl",
            ))

            # 25-75 band
            fig_fan.add_trace(go.Scatter(
                x=months, y=mc_result.percentile_bands[75] * 100,
                mode="lines", line=dict(width=0), showlegend=False,
            ))
            fig_fan.add_trace(go.Scatter(
                x=months, y=mc_result.percentile_bands[25] * 100,
                mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor="rgba(27,79,130,0.25)",
                name="25-75th pctl",
            ))

            # Median
            fig_fan.add_trace(go.Scatter(
                x=months, y=mc_result.median_path * 100,
                mode="lines", line=dict(color=BLUE, width=2),
                name="Median",
            ))

            fig_fan.add_hline(y=0, line_dash="dash", line_color="black", line_width=0.5)
            fig_fan.update_layout(**_layout("Simulated Cumulative Return Fan Chart", "Cumulative Return (%)"))
            fig_fan.update_xaxes(title_text="Month")
            st.plotly_chart(fig_fan, use_container_width=True)

        # ── Return distribution ───────────────────────────────────
        with mc2:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=mc_result.terminal_returns * 100,
                nbinsx=50, marker_color=BLUE, opacity=0.7,
                name="Terminal Return",
            ))

            for _, row in mc_result.risk_metrics.iterrows():
                fig_hist.add_vline(
                    x=row["VaR"] * 100,
                    line_dash="dash", line_color=RED, line_width=1.5,
                    annotation_text=f"VaR {row['Confidence']}",
                    annotation_position="top",
                )

            fig_hist.add_vline(x=0, line_dash="solid", line_color="black", line_width=0.5)
            fig_hist.update_layout(**_layout(
                f"Terminal Return Distribution ({mc_result.horizon_months}mo)", "Count"))
            fig_hist.update_xaxes(title_text="Cumulative Return (%)")
            st.plotly_chart(fig_hist, use_container_width=True)

        # ── Risk metrics table ────────────────────────────────────
        with mc3:
            display_df = mc_result.risk_metrics.copy()
            display_df["VaR"] = display_df["VaR"].apply(lambda x: f"{x * 100:+.2f}%")
            display_df["CVaR"] = display_df["CVaR"].apply(lambda x: f"{x * 100:+.2f}%")
            display_df["P(Loss)"] = display_df["P(Loss)"].apply(lambda x: f"{x:.1%}")
            st.dataframe(display_df.set_index("Confidence"), use_container_width=True)

            tr = mc_result.terminal_returns
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Median Return", f"{np.median(tr) * 100:+.2f}%")
            s2.metric("Mean Return", f"{np.mean(tr) * 100:+.2f}%")
            s3.metric("Std Dev", f"{np.std(tr) * 100:.2f}%")
            s4.metric("P(Loss)", f"{(tr < 0).mean():.1%}")


# ══════════════════════════════════════════════════════════════════
# TAB 6: ASK THE MODEL (explainer agent)
# ══════════════════════════════════════════════════════════════════
with tab6:
    from credit_portfolio.analytics.explainer import ModelState, explain_current_state, answer_question

    def _build_model_state() -> ModelState:
        """Build ModelState from all session state results."""
        state = ModelState()

        # Regime / BL
        hmm_bl = st.session_state.get("hmm_bl")
        if hmm_bl:
            ri = hmm_bl["regime_info"]
            state.regime = ri.get("regime", "")
            prob_key = f"p_{state.regime.lower()}"
            state.regime_confidence = ri.get(prob_key, 0.0)

            bl = hmm_bl["bl"]
            buckets = ["AAA", "AA", "A", "BBB"]
            assets = ["oas_aaa", "oas_aa", "oas_a", "oas_bbb"]
            state.prior_returns = {b: float(bl["Pi"][i]) for i, b in enumerate(buckets)}
            state.view_returns = {b: float(bl["Q_vec"][i]) for i, b in enumerate(buckets)}
            state.posterior_returns = {b: float(bl["mu_BL"][i]) for i, b in enumerate(buckets)}

            if hmm_bl["hmm"].transition_matrix is not None:
                state.transition_matrix = hmm_bl["hmm"].transition_matrix

        # Prophet
        prophet = st.session_state.get("prophet_result")
        if prophet:
            state.prophet_forecasts = prophet

        # Backtest
        hist = st.session_state.get("hist_result")
        if hist:
            state.backtest_stats = hist.stats
            # Latest weights
            if not hist.weights_history.empty:
                last_w = hist.weights_history.iloc[-1]
                state.current_weights = last_w.to_dict()
            state.benchmark_weights = {"AAA": 0.04, "AA": 0.12, "A": 0.34, "BBB": 0.50}

        # Stress
        stress = st.session_state.get("stress_result")
        if stress:
            state.stress_scenario = stress.scenario_name
            state.stress_price_impact = stress.price_impact

        # Monte Carlo
        mc = st.session_state.get("mc_result")
        if mc:
            for _, row in mc.risk_metrics.iterrows():
                if row["Confidence"] == "95%":
                    state.mc_var_95 = row["VaR"]
                    state.mc_cvar_95 = row["CVaR"]
            state.mc_p_loss = float((mc.terminal_returns < 0).mean())
            state.mc_median_return = float(np.median(mc.terminal_returns))

        return state

    st.subheader("Ask the Model")
    st.caption("Get plain-English explanations of what the model is doing and why.")

    # Full state explanation button
    if st.button("Explain Current State", use_container_width=True):
        state = _build_model_state()
        explanation = explain_current_state(state)
        st.markdown(explanation)

    st.divider()

    # Q&A interface
    user_question = st.text_input(
        "Ask a question about the model:",
        placeholder="e.g., What regime are we in? How does Black-Litterman work? What's the VaR?",
    )
    if user_question:
        state = _build_model_state()
        answer = answer_question(user_question, state)
        st.markdown(answer)
