"""
ML-based credit factor model (Amundi paper methodology).

Tree-based models predict bond-level excess returns from factor Z-scores.
SHAP values decompose predictions into time-varying factor contributions,
replacing fixed linear factor weights with data-driven nonlinear weights.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from scipy.stats import spearmanr
import shap

from credit_portfolio.data.constants import (
    ML_FEATURES, ML_MIN_TRAIN_MONTHS,
    ML_PURGE_GAP_MONTHS,
    ML_MODEL_TYPE, ML_RF_N_ESTIMATORS, ML_RF_MAX_DEPTH,
    ML_RF_MIN_SAMPLES_LEAF, ML_RF_MAX_FEATURES,
    ML_GB_N_ESTIMATORS, ML_GB_MAX_DEPTH, ML_GB_LEARNING_RATE,
    ML_GB_SUBSAMPLE, ML_GB_MIN_SAMPLES_LEAF,
    ML_ERF_OVERSAMPLE_QUINTILES, ML_ERF_OVERSAMPLE_FACTOR,
    ML_BL_BLEND_WEIGHT,
    ML_RANDOM_SEED, ML_MIN_TRAIN_SAMPLES, ML_MIN_TEST_SAMPLES,
    ML_MIN_QUINTILE_SIZE, ML_NUMERIC_TOL, ML_TARGET_COL,
)


@dataclass
class WalkForwardFold:
    """One fold of walk-forward cross-validation results."""
    train_start: str
    train_end: str
    test_period: str
    n_train: int
    n_test: int
    r2_oos: float
    ic_rank: float
    feature_importance: dict
    shap_values: np.ndarray
    predictions: np.ndarray
    actuals: np.ndarray


@dataclass
class MLFactorResult:
    """Complete result from ML factor model training and prediction."""
    model_type: str
    predictions: pd.Series
    shap_factor_weights: dict
    shap_values_current: np.ndarray
    feature_names: list
    walk_forward_folds: list
    oos_r2_mean: float
    oos_ic_mean: float
    shap_weights_history: pd.DataFrame
    regime: str


def build_model(
    model_type: str = ML_MODEL_TYPE,
    random_state: int = ML_RANDOM_SEED,
    *,
    n_estimators: int | None = None,
    max_depth: int | None = None,
    learning_rate: float | None = None,
):
    """Factory function to create the specified tree-based model.

    Optional keyword args override the default hyperparameters from constants.
    """
    if model_type == "random_forest" or model_type == "enhanced_rf":
        return RandomForestRegressor(
            n_estimators=n_estimators or ML_RF_N_ESTIMATORS,
            max_depth=max_depth or ML_RF_MAX_DEPTH,
            min_samples_leaf=ML_RF_MIN_SAMPLES_LEAF,
            max_features=ML_RF_MAX_FEATURES,
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=n_estimators or ML_GB_N_ESTIMATORS,
            max_depth=max_depth or ML_GB_MAX_DEPTH,
            learning_rate=learning_rate or ML_GB_LEARNING_RATE,
            subsample=ML_GB_SUBSAMPLE,
            min_samples_leaf=ML_GB_MIN_SAMPLES_LEAF,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def _oversample_extremes(
    X: np.ndarray,
    y: np.ndarray,
    quintiles_to_oversample: list = ML_ERF_OVERSAMPLE_QUINTILES,
    factor: float = ML_ERF_OVERSAMPLE_FACTOR,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Enhanced RF: oversample extreme return quintiles to address
    class imbalance (more normal returns than extreme ones).
    """
    if len(y) < ML_MIN_QUINTILE_SIZE:
        return X, y

    quintile_labels = pd.qcut(y, 5, labels=[1, 2, 3, 4, 5], duplicates="drop")

    X_parts = [X]
    y_parts = [y]

    for q in quintiles_to_oversample:
        mask = quintile_labels == q
        n_reps = int(factor) - 1
        if n_reps > 0 and mask.sum() > 0:
            X_parts.extend([X[mask]] * n_reps)
            y_parts.extend([y[mask]] * n_reps)

    return np.vstack(X_parts), np.concatenate(y_parts)


def walk_forward_cv(
    panel: pd.DataFrame,
    features: list = ML_FEATURES,
    target: str = ML_TARGET_COL,
    model_type: str = ML_MODEL_TYPE,
    min_train_months: int = ML_MIN_TRAIN_MONTHS,
    purge_gap: int = ML_PURGE_GAP_MONTHS,
) -> list[WalkForwardFold]:
    """
    Walk-forward cross-validation with purging and embargo.

    Expanding window: train on all data up to t-gap, predict at t.
    Purge gap prevents data leakage from overlapping return horizons.
    """
    dates = sorted(panel["date"].unique())
    folds = []

    for t_idx in range(min_train_months + purge_gap, len(dates)):
        train_end_idx = t_idx - purge_gap - 1
        test_idx = t_idx

        if train_end_idx < min_train_months - 1:
            continue

        train_dates = dates[:train_end_idx + 1]
        test_date = dates[test_idx]

        train_mask = panel["date"].isin(train_dates)
        test_mask = panel["date"] == test_date

        avail = [f for f in features if f in panel.columns]
        X_train = panel.loc[train_mask, avail].values
        y_train = panel.loc[train_mask, target].values
        X_test = panel.loc[test_mask, avail].values
        y_test = panel.loc[test_mask, target].values

        if len(X_test) == 0 or len(X_train) < ML_MIN_TRAIN_SAMPLES:
            continue

        # Remove NaN rows
        valid_train = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
        valid_test = ~(np.isnan(X_test).any(axis=1) | np.isnan(y_test))
        X_train, y_train = X_train[valid_train], y_train[valid_train]
        X_test, y_test = X_test[valid_test], y_test[valid_test]

        if len(X_train) < ML_MIN_TRAIN_SAMPLES or len(X_test) < ML_MIN_TEST_SAMPLES:
            continue

        model = build_model(model_type)

        if model_type == "enhanced_rf":
            X_fit, y_fit = _oversample_extremes(X_train, y_train)
        else:
            X_fit, y_fit = X_train, y_train

        model.fit(X_fit, y_fit)
        y_pred = model.predict(X_test)

        # Out-of-sample R-squared
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - y_test.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > ML_NUMERIC_TOL else 0.0

        # Rank IC (Spearman correlation)
        ic, _ = spearmanr(y_pred, y_test)

        feat_imp = dict(zip(avail, model.feature_importances_))

        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_test)

        folds.append(WalkForwardFold(
            train_start=str(train_dates[0]),
            train_end=str(train_dates[-1]),
            test_period=str(test_date),
            n_train=len(X_fit),
            n_test=len(X_test),
            r2_oos=float(r2),
            ic_rank=float(ic) if not np.isnan(ic) else 0.0,
            feature_importance=feat_imp,
            shap_values=shap_vals,
            predictions=y_pred,
            actuals=y_test,
        ))

    return folds


def compute_shap_factor_weights(
    shap_values: np.ndarray,
    feature_names: list,
) -> dict[str, float]:
    """
    Derive factor weights from SHAP values.

    Weight_i = mean(|SHAP_i|) / sum(mean(|SHAP_j|) for all j)
    Replaces fixed equal weights with data-driven weights.
    """
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    total = mean_abs_shap.sum()

    if total < ML_NUMERIC_TOL:
        n = len(feature_names)
        return {f: 1.0 / n for f in feature_names}

    weights = mean_abs_shap / total
    return dict(zip(feature_names, weights))


def shap_weights_over_time(
    folds: list[WalkForwardFold],
    feature_names: list,
) -> pd.DataFrame:
    """Build time series of SHAP-derived factor weights across CV folds."""
    rows = []
    for fold in folds:
        w = compute_shap_factor_weights(fold.shap_values, feature_names)
        w["date"] = fold.test_period
        rows.append(w)

    if not rows:
        return pd.DataFrame(columns=feature_names)

    df = pd.DataFrame(rows).set_index("date")
    df.index = pd.to_datetime(df.index)
    return df


def train_and_predict(
    panel: pd.DataFrame,
    current_universe: pd.DataFrame,
    features: list = ML_FEATURES,
    target: str = ML_TARGET_COL,
    model_type: str = ML_MODEL_TYPE,
    regime: str = "NORMAL",
) -> MLFactorResult:
    """
    Full ML pipeline: walk-forward CV + final model + current prediction + SHAP.
    """
    avail_features = [f for f in features if f in panel.columns
                      and f in current_universe.columns]

    # Walk-forward CV for diagnostics
    folds = walk_forward_cv(panel, avail_features, target, model_type)

    # Train final model on all historical data
    valid_rows = panel[avail_features + [target]].dropna()
    X_all = valid_rows[avail_features].values
    y_all = valid_rows[target].values

    final_model = build_model(model_type)
    if model_type == "enhanced_rf":
        X_fit, y_fit = _oversample_extremes(X_all, y_all)
    else:
        X_fit, y_fit = X_all, y_all
    final_model.fit(X_fit, y_fit)

    # Predict on current universe
    X_current = current_universe[avail_features].values
    predictions = final_model.predict(X_current)
    pred_series = pd.Series(
        predictions,
        index=current_universe["bond_id"].values,
        name="ml_expected_return",
    )

    # SHAP on current universe
    explainer = shap.TreeExplainer(final_model)
    shap_current = explainer.shap_values(X_current)
    shap_weights = compute_shap_factor_weights(shap_current, avail_features)

    # SHAP weight history
    shap_history = shap_weights_over_time(folds, avail_features)

    oos_r2s = [f.r2_oos for f in folds]
    oos_ics = [f.ic_rank for f in folds]

    return MLFactorResult(
        model_type=model_type,
        predictions=pred_series,
        shap_factor_weights=shap_weights,
        shap_values_current=shap_current,
        feature_names=avail_features,
        walk_forward_folds=folds,
        oos_r2_mean=float(np.mean(oos_r2s)) if oos_r2s else 0.0,
        oos_ic_mean=float(np.mean(oos_ics)) if oos_ics else 0.0,
        shap_weights_history=shap_history,
        regime=regime,
    )


def blend_with_bl(
    ml_predictions: pd.Series,
    bl_expected_returns: np.ndarray,
    bond_ids: np.ndarray,
    blend_weight: float = ML_BL_BLEND_WEIGHT,
) -> np.ndarray:
    """
    Blend ML predictions with BL expected returns.

    blended = (1 - w) * BL + w * ML  (scale-normalized)
    """
    ml_aligned = ml_predictions.reindex(bond_ids).fillna(0.0).values

    # Normalize ML predictions to BL scale
    if np.std(ml_aligned) > ML_NUMERIC_TOL:
        ml_scaled = (
            (ml_aligned - ml_aligned.mean()) / ml_aligned.std()
            * np.std(bl_expected_returns) + np.mean(bl_expected_returns)
        )
    else:
        ml_scaled = ml_aligned

    return (1 - blend_weight) * bl_expected_returns + blend_weight * ml_scaled
