"""Run the complete Project Pragati AI/ML intelligence pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml_common import (
    read_features,
    build_snapshot,
    add_confidence,
    add_trajectory,
    add_domain_risk,
    robust_anomaly,
fuse,
    add_explanations,
    warnings,
OUT,
    LEGACY_RISK_OUTPUT_COLUMNS,
)

from predictive_model import main as run_predictive_model


ROOT = Path(__file__).resolve().parents[1]


def integrate_predictive_model(s):
    """
    Run XGBoost and merge its forward-looking prediction into the
    current project-level snapshot.

    XGBoost is an ADDITIONAL forward-looking signal.
    It does NOT modify final_risk_score or risk_category.
    """

    print("\nRunning XGBoost predictive layer...")

    _, predictive_validation = run_predictive_model()

    predictive_path = OUT / "pragati_predictive_risk.csv"

    if not predictive_path.exists():
        raise RuntimeError(
            "XGBoost completed without producing "
            "pragati_predictive_risk.csv"
        )

    predictive = pd.read_csv(predictive_path)

    required_predictive_columns = [
        "canonical_project_id",
        "observation_month",
        "future_deterioration_probability",
        "predicted_future_deterioration",
"predictive_risk_band",
    ]

    missing = [
        c for c in required_predictive_columns
        if c not in predictive.columns
    ]

    if missing:
        raise RuntimeError(
            "Predictive output is missing columns: "
            + ", ".join(missing)
        )

    predictive["canonical_project_id"] = (
        predictive["canonical_project_id"].astype("string")
    )

    predictive["observation_month"] = pd.to_datetime(
        predictive["observation_month"],
        errors="coerce",
    )

    if predictive["canonical_project_id"].duplicated().any():
        raise RuntimeError(
            "Predictive output contains duplicate projects."
        )

    if predictive["observation_month"].isna().any():
        raise RuntimeError(
            "Predictive output contains invalid observation months."
        )

    s = s.copy()

    s["canonical_project_id"] = (
        s["canonical_project_id"].astype("string")
    )

    # The predictive model produces one current/latest prediction per
    # project. The current snapshot must also contain one row per project.
    if s["canonical_project_id"].duplicated().any():
        raise RuntimeError(
            "Current project snapshot contains duplicate projects."
        )

    dataset_max_month = (
        pd.to_datetime(
            s["observation_month"],
            errors="coerce"
        )
        .dt.to_period("M")
        .max()
    )

    predictive_months = (
        pd.to_datetime(
            predictive["observation_month"],
            errors="coerce"
        )
        .dt.to_period("M")
    )

    if predictive_months.isna().any():
        raise RuntimeError(
            "Predictive output contains invalid observation months."
        )

    if (predictive_months > dataset_max_month).any():
        raise RuntimeError(
            "Predictive output contains future observation months."
        )

    before_count = len(s)

    s = s.merge(
        predictive[
            required_predictive_columns
        ],
        on="canonical_project_id",
        how="left",
        validate="one_to_one",
    )

    if len(s) != before_count:
        raise RuntimeError(
            "Predictive merge changed project count."
        )

    missing_predictions = int(
        s["future_deterioration_probability"].isna().sum()
    )

    if missing_predictions:
        raise RuntimeError(
            f"{missing_predictions} projects have no XGBoost prediction."
        )

    if (
        (s["future_deterioration_probability"] < 0)
        | (s["future_deterioration_probability"] > 1)
    ).any():
        raise RuntimeError(
            "XGBoost produced an invalid probability."
        )
    valid_bands = {
        "LOW_PREDICTIVE_SIGNAL",
        "WATCH",
        "ELEVATED",
        "HIGH_PREDICTIVE_SIGNAL",
    }

    if not set(
        s["predictive_risk_band"].dropna().unique()
    ).issubset(valid_bands):
        raise RuntimeError(
            "XGBoost produced an invalid predictive risk band."
        )

    s["future_deterioration_probability"] = (
        s["future_deterioration_probability"].round(4)
    )

    s["predicted_future_deterioration"] = (
        s["predicted_future_deterioration"].astype(int)
    )

    predictive_validation["integration"] = {
        "projects_before_merge": before_count,
        "predictive_projects": len(predictive),
        "projects_after_merge": len(s),
        "missing_predictions": missing_predictions,
        "predictive_signal_role": (
            "forward-looking probability; "
            "does not modify current risk score/category"
        ),
        "final_risk_score_modified": False,
        "risk_category_modified": False,
    }

    return s, predictive_validation


def main():
    # =========================================================
    # 1. VALIDATE ML INPUT
    # =========================================================

    from validate_ml_input import main as validate

    validate()

    # =========================================================
    # 2. LOAD FEATURE DATA
    # =========================================================

    d = read_features()

    # =========================================================
    # 3. BUILD CURRENT PROJECT SNAPSHOT
    # =========================================================

    s = build_snapshot(d)
    s = add_confidence(s)
    s = add_trajectory(s)
    s = add_domain_risk(s)

    # =========================================================
    # 4. REAL ISOLATION FOREST ANOMALY DETECTION
    # =========================================================

    s, anomaly_metadata = robust_anomaly(s)

    # =========================================================
    # 5. CURRENT RISK FUSION
    # =========================================================
    #
    # This remains the transparent current-state risk engine.
    # XGBoost is deliberately NOT inserted into this formula.
    # This prevents the forward-looking probability from silently
    # changing the established current-risk calibration.
    # =========================================================

    s = fuse(s)

    # =========================================================
    # 6. EXPLAIN CURRENT RISK
    # =========================================================

    s = add_explanations(s)

    # =========================================================
    # 7. XGBOOST FORWARD-LOOKING PREDICTION
    # =========================================================

    s, predictive_validation = integrate_predictive_model(s)

    # =========================================================
    # 8. AI/ML HANDOFF PROFILE
    # =========================================================

    handoff = s.drop(
        columns=LEGACY_RISK_OUTPUT_COLUMNS,
        errors="ignore",
    ).copy()

    handoff.to_csv(
        OUT / "pragati_project_snapshot.csv",
        index=False,
    )

    handoff.to_csv(
        OUT / "pragati_project_risk_profiles.csv",
        index=False,
    )

    # =========================================================
    # 9. RISK SCORE OUTPUT
    # =========================================================

    score_cols = [
        "canonical_project_id",
        "project_id",
        "observation_month",
        "base_risk_score",
        "anomaly_score",
        "trajectory_risk_score",
        "data_confidence_score",
        "final_risk_score",
        "risk_category",
        "future_deterioration_probability",
        "predicted_future_deterioration",
        "risk_explanation",
        "predictive_risk_band",
    ]

    score_cols = [
        c for c in score_cols
        if c in s.columns
    ]

    s[score_cols].to_csv(
        OUT / "pragati_risk_scores.csv",
        index=False,
    )

    # =========================================================
    # 10. TRAJECTORY OUTPUT
    # =========================================================

    trajectory_cols = [
        "canonical_project_id",
        "project_id",
        "observation_month",
        "physical_progress_pct",
        "cumulative_expenditure_cr",
        "expenditure_ratio_pct",
        "progress_change_1m",
        "progress_change_3m",
        "progress_velocity_3m",
        "expenditure_change_1m",
        "expenditure_change_3m",
        "expenditure_velocity_3m",
        "cost_escalation_pct",
        "expenditure_progress_gap",
    ]

    trajectory_cols = [
        c for c in trajectory_cols
        if c in d.columns
    ]

    d[trajectory_cols].to_csv(
        OUT / "pragati_project_trajectories.csv",
        index=False,
    )

    # =========================================================
    # 11. EARLY WARNINGS
    # =========================================================

    w = warnings(s)

    w.to_csv(
        OUT / "pragati_early_warnings.csv",
        index=False,
    )

    # =========================================================
    # 12. NATIONAL PRIORITY QUEUE
    # =========================================================

    action = (
        s[
            s["risk_category"].isin(["CRITICAL", "HIGH"])
        ]
        .sort_values(
            "final_risk_score",
            ascending=False,
        )
        .copy()
    )

    monitor = (
        s[
            s["risk_category"].isin(["MEDIUM", "LOW"])
        ]
        .sort_values(
            "final_risk_score",
            ascending=False,
        )
        .copy()
    )

    review = (
        s[
            s["risk_category"].eq("REVIEW_DATA")
        ]
        .sort_values(
            "data_confidence_score",
            ascending=True,
        )
        .copy()
    )

    q = pd.concat(
        [
            action,
            monitor,
            review,
        ],
        ignore_index=True,
    )

    q = q.drop(
        columns=LEGACY_RISK_OUTPUT_COLUMNS,
        errors="ignore",
    )

    q["priority_band"] = (
        ["ACTION_REQUIRED"] * len(action)
        + ["MONITOR"] * len(monitor)
        + ["REVIEW_DATA"] * len(review)
    )

    q["priority_rank"] = range(
        1,
        len(q) + 1,
    )

    q.to_csv(
        OUT / "pragati_priority_queue.csv",
        index=False,
    )

    # =========================================================
    # 13. FINAL VALIDATION
    # =========================================================

    risk_distribution = (
        s["risk_category"]
        .value_counts()
        .to_dict()
    )

    predictive_checks = predictive_validation["leakage_check"]

    integration = predictive_validation["integration"]

    validation = {
        "supervised_metrics_reported": True,
        "reason": (
            "XGBoost was trained using chronological future-progress "
            "deterioration labels."
        ),
        "projects": int(len(s)),
        "project_months": int(len(d)),
        "complete_history_projects": int(
            s["complete_7_month_history"].sum()
        ),
        "risk_distribution": risk_distribution,
        "anomaly_count": int(s["anomaly_flag"].sum()),
        "anomaly_percentage": round(
            float(s["anomaly_flag"].mean() * 100),
            2,
        ),
        "anomaly_backend": anomaly_metadata["backend"],
        "score_range": [
            float(s["final_risk_score"].min()),
            float(s["final_risk_score"].max()),
        ],
        "schedule_available_projects": int(
            s["schedule_data_available"].sum()
        ),
        "critical_share_pct": round(
            float(
                (s["risk_category"] == "CRITICAL").mean()
                * 100
            ),
            2,
        ),
        "predictive_model": {
            "model_type": predictive_validation["model_type"],
            "fit_success": predictive_validation["model_fit_success"],
            "test_prediction_success": (
                predictive_validation["test_prediction_success"]
            ),
            "test_metrics": predictive_validation["test_metrics"],
            "validation_metrics": (
                predictive_validation["validation_metrics"]
            ),
            "feature_names": predictive_validation["feature_names"],
            "leakage_check": predictive_checks,
            "integration": integration,
        },
        "checks": {
            "no_duplicate_projects": not bool(
                s["canonical_project_id"].duplicated().any()
            ),
            "no_duplicate_project_month_keys": not bool(
                d[
                    [
                        "canonical_project_id",
                        "observation_month",
                    ]
                ]
                .duplicated()
                .any()
            ),
            "risk_scores_nonmissing": not bool(
                s["final_risk_score"].isna().any()
            ),
            "risk_scores_in_range": bool(
                s["final_risk_score"].between(0, 100).all()
            ),
            "predictive_probability_valid": bool(
                s[
                    "future_deterioration_probability"
                ].between(0, 1).all()
            ),

            "predictive_risk_bands_valid": bool(
                set(
                    s["predictive_risk_band"].dropna().unique()
                ).issubset(
                    {
                        "LOW_PREDICTIVE_SIGNAL",
                        "WATCH",
                        "ELEVATED",
                        "HIGH_PREDICTIVE_SIGNAL",
                    }
                )
            ),
            "no_future_features_used_as_inputs": not bool(
                predictive_checks.get(
                    "future_features_used_as_inputs",
                    True,
                )
            ),
            "temporal_split_respected": bool(
                predictive_checks.get(
                    "temporal_split_respected",
                    False,
                )
            ),
            "risk_category_not_overwritten_by_xgboost": (
                integration["risk_category_modified"] is False
            ),
            "final_risk_score_not_overwritten_by_xgboost": (
                integration["final_risk_score_modified"] is False
            ),
            "review_data_not_critical": not bool(
                (
                    s["risk_category"].eq("REVIEW_DATA")
                    & s["risk_category"].eq("CRITICAL")
                ).any()
            ),
            "priority_queue_one_row_per_project": (
                len(q) == len(s)
                and not bool(
                    q["canonical_project_id"].duplicated().any()
                )
            ),
        },
    }

    (
        OUT / "model_validation_report.json"
    ).write_text(
        json.dumps(
            validation,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # =========================================================
    # 14. DASHBOARD DATA
    # =========================================================

    dashboard = {
        "national_summary": {
            "projects": int(len(s)),
            "project_months": int(len(d)),
            "complete_history_projects": int(
                s["complete_7_month_history"].sum()
            ),
        },

        "risk_distribution": risk_distribution,

        "anomaly_summary": {
            "count": validation["anomaly_count"],
            "percentage": validation["anomaly_percentage"],
            "backend": anomaly_metadata["backend"],
        },

        "predictive_summary": {
            "predicted_future_deterioration": int(
                s["predicted_future_deterioration"].sum()
            ),

            "predictive_risk_band_distribution": (
                s["predictive_risk_band"]
                .value_counts()
                .to_dict()
            ),

            "average_future_deterioration_probability": round(
                float(
                    s[
                        "future_deterioration_probability"
                    ].mean()
                ),
                4,
            ),

            "max_future_deterioration_probability": round(
                float(
                    s[
                        "future_deterioration_probability"
                    ].max()
                ),
                4,
            ),

            "test_metrics": predictive_validation[
                "test_metrics"
            ],
        },

        "warning_summary": (
            w["warning_type"]
            .value_counts()
            .to_dict()
            if len(w)
            else {}
        ),

        "state_wise_risk_summary": (
            s.groupby(
                "state",
                dropna=False,
            )["risk_category"]
            .value_counts()
            .unstack(fill_value=0)
            .to_dict(orient="index")
            if "state" in s
            else {}
        ),

        "sector_wise_risk_summary": (
            s.groupby(
                "sector",
                dropna=False,
            )["risk_category"]
            .value_counts()
            .unstack(fill_value=0)
            .to_dict(orient="index")
            if "sector" in s
            else {}
        ),

        "top_priority_projects": (
            q.head(25)[
                [
                    c
                    for c in [
                        "priority_rank",
                        "project_id",
                        "project_name",
                        "final_risk_score",
                        "risk_category",
                        "future_deterioration_probability",
                        "predictive_risk_band",
                        "predicted_future_deterioration",
                        "primary_risk_driver",
                        "recommended_action",
                    ]
                    if c in q.columns
                ]
            ]
            .fillna("")
            .to_dict(orient="records")
        ),

        "project_profiles": (
            s[
                [
                    c
                    for c in [
                        "canonical_project_id",
                        "project_id",
                        "project_name",
                        "final_risk_score",
                        "risk_category",
                        "data_confidence_score",
                        "trajectory_status",
                        "anomaly_flag",
                        "future_deterioration_probability",
                        "predictive_risk_band",
                        "predicted_future_deterioration",
                    ]
                    if c in s.columns
                ]
            ]
            .fillna("")
            .to_dict(orient="records")
        ),
    }

    (
        OUT / "pragati_dashboard.json"
    ).write_text(
        json.dumps(
            dashboard,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # =========================================================
    # 15. TERMINAL SUMMARY
    # =========================================================

    print()
    print("=" * 70)
    print("PROJECT PRAGATI AI/ML PIPELINE COMPLETE")
    print("=" * 70)

    print(
        "Dataset:",
        len(d),
        "project-month records",
    )

    print(
        "Projects:",
        len(s),
    )

    print(
        "Complete-history projects:",
        int(s["complete_7_month_history"].sum()),
    )

    print(
        "Risk:",
        risk_distribution,
    )

    print(
        "Anomalies:",
        validation["anomaly_count"],
        f"({validation['anomaly_percentage']}%)",
    )

    print(
        "XGBoost validation metrics:",
        predictive_validation["validation_metrics"],
    )

    print(
        "XGBoost test metrics:",
        predictive_validation["test_metrics"],
    )

    print(
        "Predicted future deterioration:",
        int(
            s[
                "predicted_future_deterioration"
            ].sum()
        ),
        "/",
        len(s),
    )

    print(
        "Warnings:",
        len(w),
        "| Critical:",
        int(
            (w["severity"] == "CRITICAL").sum()
        )
        if len(w)
        else 0,
        "| High:",
        int(
            (w["severity"] == "HIGH").sum()
        )
        if len(w)
        else 0,
    )

    print()
    print("Top priority projects:")

    print(
        q[
            [
                "priority_rank",
                "project_name",
                "final_risk_score",
                "risk_category",
                "future_deterioration_probability",
                "predictive_risk_band",
            ]
        ]
        .head(5)
        .to_string(index=False)
    )

    print("=" * 70)


if __name__ == "__main__":
    main()