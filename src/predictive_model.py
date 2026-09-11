"""Temporal XGBoost prediction layer for Project Pragati."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FEATURES = OUT / "pragati_features.csv"
MODEL_PATH = ROOT / "models" / "pragati_xgboost_model.joblib"


FEATURE_CANDIDATES = [
    "cost_escalation_pct",
    "expenditure_ratio_pct",
    "expenditure_progress_gap",
    "physical_progress_pct",
    "progress_change_1m",
    "progress_change_3m",
    "progress_change_6m",
    "progress_velocity_3m",
    "expenditure_change_1m",
    "expenditure_change_3m",
    "expenditure_velocity_3m",
    "project_age_months",
]


def load_data():
    d = pd.read_csv(FEATURES)

    d["observation_month"] = pd.to_datetime(
        d["observation_month"],
        errors="coerce",
    )

    d["canonical_project_id"] = (
        d["canonical_project_id"].astype("string")
    )

    d = d.sort_values(
        ["canonical_project_id", "observation_month"],
        kind="stable",
    ).reset_index(drop=True)

    return d


def make_month_pair(d, start_month, end_month):
    """Create supervised samples from month t to month t+1."""

    start = d[
        d["observation_month"].dt.strftime("%Y-%m").eq(start_month)
    ].copy()

    end = d[
        d["observation_month"].dt.strftime("%Y-%m").eq(end_month)
    ].copy()

    if start.empty or end.empty:
        return pd.DataFrame()

    feature_cols = [
        c
        for c in FEATURE_CANDIDATES
        if c in start.columns
    ]

    left_columns = [
        "canonical_project_id",
        "project_id",
        "project_name",
        "observation_month",
    ] + feature_cols

    left = start[left_columns].copy()

    right = end[
        [
            "canonical_project_id",
            "physical_progress_pct",
        ]
    ].copy()

    right = right.rename(
        columns={
            "physical_progress_pct":
                "future_physical_progress_pct"
        }
    )

    pair = left.merge(
        right,
        on="canonical_project_id",
        how="inner",
        validate="one_to_one",
    )

    pair["target"] = np.where(
        pair["physical_progress_pct"].notna()
        & pair["future_physical_progress_pct"].notna(),
        (
            pair["future_physical_progress_pct"]
            < pair["physical_progress_pct"]
        ).astype(int),
        np.nan,
    )

    pair = pair[
        pair["target"].notna()
    ].copy()

    pair["target"] = pair["target"].astype(int)

    pair["prediction_period"] = (
        start_month + "_to_" + end_month
    )

    # IMPORTANT: this return MUST remain inside this function.
    return pair


def make_datasets(d):
    """Build chronological train, validation and test datasets."""

    train_periods = [
        ("2026-01", "2026-02"),
        ("2026-02", "2026-03"),
        ("2026-03", "2026-04"),
        ("2026-04", "2026-05"),
    ]

    validation_period = (
        "2026-05",
        "2026-06",
    )

    test_period = (
        "2026-06",
        "2026-07",
    )

    train_parts = []

    for start_month, end_month in train_periods:

        pair = make_month_pair(
            d,
            start_month,
            end_month,
        )

        if not pair.empty:
            train_parts.append(pair)

    if not train_parts:
        raise RuntimeError(
            "No training pairs could be constructed."
        )

    train = pd.concat(
        train_parts,
        ignore_index=True,
    )

    validation = make_month_pair(
        d,
        validation_period[0],
        validation_period[1],
    )

    test = make_month_pair(
        d,
        test_period[0],
        test_period[1],
    )

    if validation.empty:
        raise RuntimeError(
            "No validation pairs could be constructed."
        )

    if test.empty:
        raise RuntimeError(
            "No test pairs could be constructed."
        )

    return train, validation, test


def calculate_metrics(
    y_true,
    probabilities,
    predictions,
):
    """Calculate classification metrics."""

    result = {
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": None,
    }

    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        )

    return result


def main():

    # ---------------------------------------------------------
    # 1. LOAD DATA
    # ---------------------------------------------------------

    d = load_data()

    # ---------------------------------------------------------
    # 2. BUILD TEMPORAL TRAIN / VALIDATION / TEST SETS
    # ---------------------------------------------------------

    train, validation, test = make_datasets(d)

    # ---------------------------------------------------------
    # 3. SELECT FEATURES
    # ---------------------------------------------------------

    feature_cols = [
        c
        for c in FEATURE_CANDIDATES
        if c in train.columns
        and train[c].notna().any()
    ]

    if not feature_cols:
        raise RuntimeError(
            "No predictive features have usable training data."
        )

    # ---------------------------------------------------------
    # 4. TRAIN-ONLY MEDIAN IMPUTATION
    # ---------------------------------------------------------

    X_train = train[
        feature_cols
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    train_medians = X_train.median()

    # Remove features whose training median is unavailable.
    feature_cols = [
        c
        for c in feature_cols
        if pd.notna(train_medians[c])
    ]

    if not feature_cols:
        raise RuntimeError(
            "No predictive features have valid training medians."
        )

    X_train = train[
        feature_cols
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    X_validation = validation[
        feature_cols
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    X_test = test[
        feature_cols
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    train_medians = X_train.median()

    X_train = X_train.fillna(
        train_medians
    )

    X_validation = X_validation.fillna(
        train_medians
    )

    X_test = X_test.fillna(
        train_medians
    )

    if X_train.isna().any().any():
        raise RuntimeError(
            "Training features still contain missing values."
        )

    if X_validation.isna().any().any():
        raise RuntimeError(
            "Validation features still contain missing values."
        )

    if X_test.isna().any().any():
        raise RuntimeError(
            "Test features still contain missing values."
        )

    # ---------------------------------------------------------
    # 5. TARGET
    # ---------------------------------------------------------

    y_train = train["target"].astype(int)
    y_validation = validation["target"].astype(int)
    y_test = test["target"].astype(int)

    if y_train.nunique() < 2:
        raise RuntimeError(
            "Training target contains only one class."
        )

    negative = int(
        (y_train == 0).sum()
    )

    positive = int(
        (y_train == 1).sum()
    )

    scale_pos_weight = (
        negative / positive
        if positive > 0
        else 1.0
    )

    # ---------------------------------------------------------
    # 6. XGBOOST MODEL
    # ---------------------------------------------------------

    model = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )

    # ACTUAL MODEL TRAINING
    model.fit(
        X_train,
        y_train,
    )

    # ---------------------------------------------------------
    # 7. VALIDATION
    # ---------------------------------------------------------

    validation_probability = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    validation_prediction = (
        validation_probability >= 0.5
    ).astype(int)

    validation_metrics = calculate_metrics(
        y_validation,
        validation_probability,
        validation_prediction,
    )

    # ---------------------------------------------------------
    # 8. TEST
    # ---------------------------------------------------------

    test_probability = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    test_prediction = (
        test_probability >= 0.5
    ).astype(int)

    test_metrics = calculate_metrics(
        y_test,
        test_probability,
        test_prediction,
    )

    # ---------------------------------------------------------
    # 9. JULY CURRENT PROJECT PREDICTIONS
    # ---------------------------------------------------------

    latest = (
        d.sort_values(
            [
                "canonical_project_id",
                "observation_month",
            ],
            kind="stable",
        )
        .groupby(
            "canonical_project_id",
            sort=False,
        )
        .tail(1)
        .copy()
    )

    X_latest = latest[
        feature_cols
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    X_latest = X_latest.fillna(
        train_medians
    )

    latest_probability = (
        model.predict_proba(
            X_latest
        )[:, 1]
    )

    latest_prediction = (
        latest_probability >= 0.5
    ).astype(int)

    predictive_output = latest[
        [
            "canonical_project_id",
            "project_id",
            "project_name",
            "observation_month",
        ]
    ].copy()

    predictive_output[
        "future_deterioration_probability"
    ] = latest_probability

    predictive_output[
        "predicted_future_deterioration"
    ] = latest_prediction

    predictive_output[
        "predictive_risk_band"
    ] = pd.cut(
        latest_probability,
        bins=[-np.inf, 0.30, 0.50, 0.70, np.inf],
        labels=[
            "LOW_PREDICTIVE_SIGNAL",
            "WATCH",
            "ELEVATED",
            "HIGH_PREDICTIVE_SIGNAL",
            ],
        right=False,
    )

    # The continuous probability is the primary predictive signal.
    # The 0.5 binary prediction is retained as the standard classifier
    # output, but it is NOT the primary operational interpretation.
    valid_predictive_bands = {
        "LOW_PREDICTIVE_SIGNAL",
        "WATCH",
        "ELEVATED",
        "HIGH_PREDICTIVE_SIGNAL",
    }

    if not set(
        predictive_output["predictive_risk_band"].dropna().unique()
    ).issubset(valid_predictive_bands):
        raise RuntimeError(
            "Invalid predictive risk band generated."
        )

    predictive_output.to_csv(
        OUT / "pragati_predictive_risk.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 10. FEATURE IMPORTANCE
    # ---------------------------------------------------------

    importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    importance.to_csv(
        OUT / "pragati_predictive_feature_importance.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 11. SAVE MODEL
    # ---------------------------------------------------------

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "model": model,
            "features": feature_cols,
            "train_medians": train_medians.to_dict(),
            "target": (
                "1 if physical_progress_pct(t+1) "
                "< physical_progress_pct(t), else 0"
            ),
        },
        MODEL_PATH,
    )

    # ---------------------------------------------------------
    # 12. VALIDATION REPORT
    # ---------------------------------------------------------

    validation_report = {
        "model_type": "XGBClassifier",
        "implementation": "xgboost.XGBClassifier",
        "model_fit_success": True,
        "test_prediction_success": True,

        "target_definition": (
            "future_progress_deterioration = 1 when "
            "physical_progress_pct(t+1) < "
            "physical_progress_pct(t)"
        ),

        "train_period": [
            "2026-01_to_2026-02",
            "2026-02_to_2026-03",
            "2026-03_to_2026-04",
            "2026-04_to_2026-05",
        ],

        "validation_period": "2026-05_to_2026-06",
        "test_period": "2026-06_to_2026-07",

        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),

        "positive_rate_train": float(
            y_train.mean()
        ),

        "positive_rate_validation": float(
            y_validation.mean()
        ),

        "positive_rate_test": float(
            y_test.mean()
        ),

        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,

        "feature_count": len(feature_cols),
        "feature_names": feature_cols,

        "latest_prediction_rows": int(
            len(predictive_output)
        ),

        "predictive_probability_is_primary_signal": True,

        "binary_prediction_threshold": 0.5,

        "binary_prediction_is_primary_operational_signal": False,

        "predictive_risk_bands": {
            "LOW_PREDICTIVE_SIGNAL": "<0.30",
            "WATCH": "0.30-<0.50",
            "ELEVATED": "0.50-<0.70",
            "HIGH_PREDICTIVE_SIGNAL": ">=0.70",
        },

        "predictive_risk_band_distribution": (
            predictive_output["predictive_risk_band"]
            .value_counts()
            .to_dict()
        ),

        "leakage_check": {
            "future_features_used_as_inputs": False,
            "target_derived_from_future_progress_only": True,
            "risk_score_used_as_target": False,
            "risk_category_used_as_target": False,
            "train_medians_used_for_validation_and_test": True,
            "temporal_split_respected": True,
        },

        "model_parameters": {
            "n_estimators": 200,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "scale_pos_weight": scale_pos_weight,
        },

        "july_predictions_note": (
            "July predictions are forward-looking predictions "
            "for the next month. They are not evaluated because "
            "August 2026 ground truth is unavailable."
        ),

        "model_path": str(MODEL_PATH),
    }

    (
        OUT / "predictive_model_validation.json"
    ).write_text(
        json.dumps(
            validation_report,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # 13. TERMINAL SUMMARY
    # ---------------------------------------------------------

    print(
        "XGBoost: FIT SUCCESS | "
        f"train={len(train)} "
        f"validation={len(validation)} "
        f"test={len(test)}"
    )

    print(
        "Features used:",
        feature_cols,
    )

    print(
        "Validation metrics:",
        validation_metrics,
    )

    print(
        "Test metrics:",
        test_metrics,
    )

    print(
        "Predictive output:",
        len(predictive_output),
        "latest projects",
    )

    print(
        "Predictive bands:",
        predictive_output[
            "predictive_risk_band"
        ].value_counts().to_dict(),
    )

    return predictive_output, validation_report


if __name__ == "__main__":
    main()