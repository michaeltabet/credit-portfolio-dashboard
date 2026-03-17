"""
Consolidated HMM regime detection for credit spreads.

Fits a 3-state Gaussian HMM on [OAS level, 1m change, 3m change, 6m change]
to classify market regimes as COMPRESSION / NORMAL / STRESS.
"""

import numpy as np
import pandas as pd
import warnings
from dataclasses import dataclass

warnings.filterwarnings("ignore")
from hmmlearn.hmm import GaussianHMM

from credit_portfolio.data.constants import (
    REGIME_LABELS, TAU_BY_REGIME, OMEGA_SCALE, REGIME_COLORS, SERIES_MAP,
    HMM_N_STATES, HMM_MAX_ITER, HMM_RANDOM_SEED, HMM_CONVERGENCE_TOL,
    OAS_PCT_TO_BP,
)


@dataclass
class HMMResult:
    states           : pd.Series
    state_probs      : pd.DataFrame
    current_regime   : int
    current_label    : str
    omega_scale      : float
    transition_matrix: np.ndarray
    regime_stats     : pd.DataFrame
    tau              : float
    monthly          : pd.DataFrame  # enriched monthly data


def prepare_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare monthly data with derived features for HMM fitting."""
    monthly = df.copy()
    if "oas_ig_chg_1m" not in monthly.columns:
        monthly["oas_ig_chg_1m"] = monthly["oas_ig"].diff(1)
    if "oas_ig_chg_3m" not in monthly.columns:
        monthly["oas_ig_chg_3m"] = monthly["oas_ig"].diff(3)
    if "oas_hy_ig_ratio" not in monthly.columns and "oas_hy" in monthly.columns:
        monthly["oas_hy_ig_ratio"] = monthly["oas_hy"] / monthly["oas_ig"]
    for col in ["tr_ig", "tr_bbb", "tr_hy"]:
        ret_col = f"ret_{col}"
        if col in monthly.columns and ret_col not in monthly.columns:
            monthly[ret_col] = monthly[col].pct_change()
    return monthly


def fit_hmm(df: pd.DataFrame, n_states: int = HMM_N_STATES,
            n_iter: int = HMM_MAX_ITER, random_state: int = HMM_RANDOM_SEED,
            refit_every: int = 12) -> HMMResult:
    """
    Fit 3-state Gaussian HMM using an expanding window — no look-ahead.

    The model is refit every `refit_every` months on all data up to that point.
    Between refits, new observations are classified using the most recent model.
    At no point does the model see future data.

    Parameters
    ----------
    df : monthly DataFrame with at least 'oas_ig' column
    n_states : number of hidden states
    n_iter : max EM iterations
    random_state : for reproducibility
    refit_every : refit the HMM every N months (default 12 = annual refit)

    Returns
    -------
    HMMResult with states, probabilities, regime statistics, and enriched monthly data
    """
    monthly = prepare_monthly(df)

    feats = pd.DataFrame({
        "level" : monthly["oas_ig"],
        "chg_1m": monthly["oas_ig"].diff(1),
        "chg_3m": monthly["oas_ig"].diff(3),
        "chg_6m": monthly["oas_ig"].diff(6),
    }).dropna()

    n_obs = len(feats)
    min_fit = max(36, n_states * 12)  # need at least 36 months to fit

    # Storage for per-step results
    all_states = np.full(n_obs, -1, dtype=int)
    all_probs = np.zeros((n_obs, n_states))

    # Track current model + normalization params
    cur_model = None
    cur_mapping = None
    cur_mu = None
    cur_sd = None

    for t in range(n_obs):
        if t + 1 < min_fit:
            continue

        # Refit at min_fit, then every refit_every months, and at the last obs
        need_refit = (
            cur_model is None
            or (t - min_fit) % refit_every == 0
            or t == n_obs - 1
        )

        if need_refit:
            window_feats = feats.iloc[:t + 1]
            mu_f = window_feats.mean()
            sd_f = window_feats.std().replace(0, 1.0)
            X_window = ((window_feats - mu_f) / sd_f).values

            model = GaussianHMM(n_components=n_states, covariance_type="full",
                                n_iter=n_iter, random_state=random_state,
                                tol=HMM_CONVERGENCE_TOL)
            try:
                model.fit(X_window)
                # Map states by OAS level (ascending = compression/normal/stress)
                oas_rank = np.argsort(model.means_[:, 0])
                mapping = {int(s): int(lbl) for lbl, s in enumerate(oas_rank)}
                cur_model = model
                cur_mapping = mapping
                cur_mu = mu_f
                cur_sd = sd_f
            except Exception:
                pass  # keep using previous model

        if cur_model is None:
            continue

        # Classify current observation using the current model
        x_t = ((feats.iloc[t:t+1] - cur_mu) / cur_sd).values
        try:
            raw = cur_model.predict(x_t)[0]
            all_states[t] = cur_mapping[int(raw)]
            p = cur_model.predict_proba(x_t)[0]
            inv_map = {v: k for k, v in cur_mapping.items()}
            all_probs[t] = [p[inv_map[lbl]] for lbl in range(n_states)]
        except Exception:
            # If prediction fails, carry forward previous state
            if t > 0 and all_states[t - 1] >= 0:
                all_states[t] = all_states[t - 1]
                all_probs[t] = all_probs[t - 1]

    # Back-fill early periods (before min_fit) with the first valid state
    first_valid = np.where(all_states >= 0)[0]
    if len(first_valid) > 0:
        fv = first_valid[0]
        all_states[:fv] = all_states[fv]
        all_probs[:fv] = all_probs[fv]

    states = pd.Series(all_states, index=feats.index, name="regime")

    probs = pd.DataFrame(
        {lbl: all_probs[:, lbl] for lbl in range(n_states)},
        index=feats.index
    )

    # Rebuild transition matrix from the final model
    T = np.zeros((n_states, n_states))
    if cur_model is not None and cur_mapping is not None:
        for ir in range(n_states):
            for jr in range(n_states):
                T[cur_mapping[ir], cur_mapping[jr]] = cur_model.transmat_[ir, jr]

    # Regime statistics
    rows = []
    for rid, label in REGIME_LABELS.items():
        mask = states == rid
        oas_v = monthly.loc[states[mask].index, "oas_ig"].dropna()
        chg_v = monthly["oas_ig"].diff(1).loc[states[mask].index].dropna()
        rows.append({
            "regime"             : label,
            "n_months"           : int(mask.sum()),
            "pct_time"           : round(mask.mean() * 100, 1),
            "mean_oas_bp"        : round(float(oas_v.mean()) * OAS_PCT_TO_BP, 0) if len(oas_v) > 0 else 0,
            "std_oas_bp"         : round(float(oas_v.std()) * OAS_PCT_TO_BP, 0) if len(oas_v) > 1 else 0,
            "mean_monthly_chg_bp": round(float(chg_v.mean()) * OAS_PCT_TO_BP, 1) if len(chg_v) > 0 else 0,
        })
    regime_stats = pd.DataFrame(rows).set_index("regime")

    current_regime = int(all_states[-1])
    current_label = REGIME_LABELS[current_regime]

    # Enrich monthly data with regime info
    monthly = monthly.copy()
    monthly["hmm_state"] = states.reindex(monthly.index)
    monthly["regime"] = monthly["hmm_state"].map(REGIME_LABELS)
    monthly["p_compression"] = probs[0].reindex(monthly.index)
    monthly["p_normal"] = probs[1].reindex(monthly.index)
    monthly["p_stress"] = probs[2].reindex(monthly.index)
    monthly["bl_tau"] = monthly["regime"].map(TAU_BY_REGIME)

    return HMMResult(
        states=states,
        state_probs=probs,
        current_regime=current_regime,
        current_label=current_label,
        omega_scale=OMEGA_SCALE[current_regime],
        transition_matrix=T,
        regime_stats=regime_stats,
        tau=TAU_BY_REGIME[current_label],
        monthly=monthly,
    )


def get_current_regime(hmm_result: HMMResult) -> dict:
    """Extract current regime info as a dict (for BL pipeline)."""
    monthly = hmm_result.monthly
    last = monthly.dropna(subset=["regime"]).iloc[-1]
    return {
        "date"         : last.name,
        "oas_ig"       : round(float(last["oas_ig"]), 3),
        "regime"       : last["regime"],
        "bl_tau"       : float(last["bl_tau"]),
        "p_compression": round(float(last.get("p_compression", 0)), 3),
        "p_normal"     : round(float(last.get("p_normal", 0)), 3),
        "p_stress"     : round(float(last.get("p_stress", 0)), 3),
    }


def regime_summary(hmm_result: HMMResult) -> pd.DataFrame:
    """Return regime statistics summary."""
    return hmm_result.regime_stats
