
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR, OUT_DIR = ROOT / "data", ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_input() -> Path:
    """Prefer the requested clean file; otherwise use the supplied master CSV."""
    preferred = DATA_DIR / "pragati_jan_jul_2026_FINAL_CLEAN.csv"
    if preferred.exists():
        return preferred
    candidates = sorted(DATA_DIR.glob("*pragati*.[cC][sS][vV]"))
    if not candidates:
        raise FileNotFoundError("No Pragati CSV found in data/.")
    return candidates[0]


def numeric(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce") if column in df else pd.Series(np.nan, index=df.index)


def robust_anomaly_by_month(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Unsupervised robust outlier score, fit separately each observation month.

    Median imputation is used only inside this anomaly calculation; it never
    changes source values or claims missing measurements are zero.
    """
    score = pd.Series(np.nan, index=df.index, dtype=float)
    for _, idx in df.groupby("observation_month", sort=False).groups.items():
        x = df.loc[idx, features].copy()
        x = x.apply(pd.to_numeric, errors="coerce")
        med = x.median(axis=0)
        mad = (x - med).abs().median(axis=0)
        usable = mad > 0
        if not usable.any():
            score.loc[idx] = 0.0
            continue
        z = (x.loc[:, usable] - med[usable]).abs().div(1.4826 * mad[usable])
        # Missing values are median-imputed solely for the model input.
        z = z.fillna(0.0).clip(upper=10)
        score.loc[idx] = z.mean(axis=1)
    df["anomaly_score"] = score
    # A fixed quantile gives an interpretable portfolio-alert rate each month.
    cutoff = df.groupby("observation_month")["anomaly_score"].transform(lambda s: s.quantile(0.95))
    df["anomaly_flag"] = (df["anomaly_score"] >= cutoff) & df["anomaly_score"].notna()
    return df


def make_reasons(row: pd.Series) -> str:
    reasons: list[str] = []
    if pd.notna(row["progress_velocity_3m"]) and row["progress_velocity_3m"] <= 0:
        reasons.append(f"3-month physical progress is stagnant/declining ({row['progress_velocity_3m']:.2f} pp/month)")
    elif pd.notna(row["progress_velocity_3m"]) and row["progress_velocity_3m"] < 0.5:
        reasons.append(f"3-month physical progress is slow ({row['progress_velocity_3m']:.2f} pp/month)")
    elif pd.notna(row["progress_change_1m"]) and row["progress_change_1m"] <= 0:
        reasons.append(f"latest monthly physical progress is non-positive ({row['progress_change_1m']:.2f} pp)")
    if bool(row["anomaly_flag"]):
        reasons.append(f"month-local anomaly signal ({row['anomaly_score']:.2f} robust deviation score)")
    if pd.isna(row["schedule_slippage_months"]):
        reasons.append("schedule dates unavailable in supplied CSV")
    if not bool(row["historical_financial_scoring_available"]):
        reasons.append("cost/imbalance excluded: revised-cost timing is ambiguous")
    return "; ".join(reasons) if reasons else "limited evidence: no trend history or anomaly signal"


def main() -> None:
    input_file = find_input()
    df = pd.read_csv(input_file)
    source_columns = df.columns.tolist()

    required = {"observation_month", "project_id", "cumulative_expenditure_cr", "physical_progress_pct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    df["observation_month"] = pd.to_datetime(df["observation_month"], errors="coerce")
    if df["observation_month"].isna().any():
        raise ValueError("Some observation_month values cannot be parsed.")
    df["canonical_project_id"] = df.get("canonical_project_id", df["project_id"]).astype("string")
    df["physical_progress_pct"] = numeric(df, "physical_progress_pct")
    df["cumulative_expenditure_cr"] = numeric(df, "cumulative_expenditure_cr")
    df["original_cost_cr"] = numeric(df, "original_cost_cr")

    # The supplied file calls this a *latest* revised cost.  It is preserved as
    # a descriptive feature, but never assumed to have been known historically.
    revised_source = "revised_cost_cr" if "revised_cost_cr" in df else "latest_revised_cost_cr"
    df["revised_cost_cr"] = numeric(df, revised_source)
    df["revised_cost_source"] = revised_source
    df["historical_financial_scoring_available"] = revised_source == "revised_cost_cr"

    denom_original = df["original_cost_cr"].where(df["original_cost_cr"] > 0)
    denom_revised = df["revised_cost_cr"].where(df["revised_cost_cr"] > 0)
    df["cost_escalation_pct"] = (df["revised_cost_cr"] - df["original_cost_cr"]).div(denom_original).mul(100)
    df["expenditure_ratio_pct"] = df["cumulative_expenditure_cr"].div(denom_revised).mul(100)
    df["expenditure_progress_gap"] = df["expenditure_ratio_pct"] - df["physical_progress_pct"]

    # Raw date-string diagnosis happens before any conversion.  If date columns
    # are absent (as in the supplied master CSV), explicitly create unavailable
    # schedule fields rather than fabricating a schedule.
    date_columns = [c for c in ("approval_date", "original_completion_date", "revised_completion_date") if c in df]
    raw_date_nonmissing = {c: int(df[c].notna().sum()) for c in date_columns}
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], format="mixed", errors="coerce")
    if {"original_completion_date", "revised_completion_date"}.issubset(df.columns):
        df["schedule_slippage_days"] = (df["revised_completion_date"] - df["original_completion_date"]).dt.days
    else:
        df["schedule_slippage_days"] = np.nan
    df["schedule_slippage_months"] = df["schedule_slippage_days"] / 30.44
    if "approval_date" in df:
        df["project_age_days"] = (df["observation_month"] - df["approval_date"]).dt.days
    else:
        df["project_age_days"] = np.nan
    df["project_age_months"] = df["project_age_days"] / 30.44

    df = df.sort_values(["canonical_project_id", "observation_month"], kind="stable").reset_index(drop=True)
    grouped = df.groupby("canonical_project_id", sort=False)
    for base, destination, periods in [
        ("physical_progress_pct", "progress_change_1m", 1),
        ("physical_progress_pct", "progress_change_3m", 3),
        ("physical_progress_pct", "progress_change_6m", 6),
        ("cumulative_expenditure_cr", "expenditure_change_1m", 1),
        ("cumulative_expenditure_cr", "expenditure_change_3m", 3),
    ]:
        df[destination] = grouped[base].diff(periods)
    df["progress_velocity_3m"] = df["progress_change_3m"] / 3
    df["expenditure_velocity_3m"] = df["expenditure_change_3m"] / 3

    # Explainable dimensions.  Weight is only used when its source is known at
    # the observation month; available weights are re-normalized, not zero-filled.
    df["cost_risk"] = (df["cost_escalation_pct"].clip(lower=0, upper=100) / 100).where(df["historical_financial_scoring_available"])
    df["schedule_risk"] = (df["schedule_slippage_months"].clip(lower=0, upper=36) / 36)
    velocity = df["progress_velocity_3m"].combine_first(df["progress_change_1m"])
    df["progress_risk"] = (1 - velocity.clip(lower=0, upper=5) / 5).where(velocity.notna())
    df["imbalance_risk"] = (df["expenditure_progress_gap"].clip(lower=0, upper=50) / 50).where(df["historical_financial_scoring_available"])
    weight_map = {"cost_risk": 0.30, "schedule_risk": 0.30, "progress_risk": 0.25, "imbalance_risk": 0.15}
    numerator = sum(df[c].fillna(0) * w for c, w in weight_map.items())
    denominator = sum(df[c].notna().astype(float) * w for c, w in weight_map.items())
    df["risk_evidence_weight"] = denominator
    df["risk_score"] = (100 * numerator / denominator.replace(0, np.nan)).clip(0, 100)
    df["risk_confidence"] = (denominator / sum(weight_map.values())).clip(0, 1)

    anomaly_features = ["physical_progress_pct", "cumulative_expenditure_cr", "progress_change_1m", "progress_change_3m", "expenditure_change_1m", "expenditure_change_3m"]
    df = robust_anomaly_by_month(df, anomaly_features)
    df["risk_level"] = pd.cut(df["risk_score"], [-np.inf, 30, 60, 80, np.inf], labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"]).astype("string")
    df.loc[df["risk_score"].isna(), "risk_level"] = "INSUFFICIENT_EVIDENCE"
    df["risk_reasons"] = df.apply(make_reasons, axis=1)

    latest = (df.sort_values(["canonical_project_id", "observation_month"], kind="stable")
                .groupby("canonical_project_id", as_index=False, sort=False).tail(1).copy())
    # Ranked score first; anomaly score is a transparent tie-breaker only.
    latest["priority_rank"] = latest["risk_score"].rank(method="min", ascending=False, na_option="bottom").astype(int)
    latest["priority_level"] = latest["risk_level"]
    latest["priority_level"] = latest["priority_level"].where(latest["risk_score"].notna(), "REVIEW_DATA")
    latest = latest.sort_values(["priority_rank", "anomaly_score"], ascending=[True, False], na_position="last")

    features_path = OUT_DIR / "pragati_features.csv"
    scores_path = OUT_DIR / "pragati_risk_scores.csv"
    queue_path = OUT_DIR / "pragati_priority_queue.csv"
    df.to_csv(features_path, index=False)
    score_columns = ["observation_month", "canonical_project_id", "project_id", "project_name", "risk_score", "risk_level", "risk_confidence", "risk_evidence_weight", "anomaly_score", "anomaly_flag", "risk_reasons"]
    df[[c for c in score_columns if c in df]].to_csv(scores_path, index=False)
    queue_columns = ["priority_rank", "priority_level", "observation_month", "canonical_project_id", "project_id", "project_name", "line_ministry", "sector", "risk_score", "risk_level", "risk_confidence", "anomaly_score", "anomaly_flag", "risk_reasons", "source_file", "source_page"]
    latest[[c for c in queue_columns if c in latest]].to_csv(queue_path, index=False)

    month_counts = df["observation_month"].dt.strftime("%Y-%m").value_counts().sort_index().to_dict()
    report = {
        "input_file": str(input_file.relative_to(ROOT)), "input_shape": list(df.shape),
        "source_column_count": len(source_columns), "unique_projects": int(df["canonical_project_id"].nunique()),
        "month_counts": month_counts, "projects_with_all_seven_months": int((df.groupby("canonical_project_id")["observation_month"].nunique() == 7).sum()),
        "duplicate_project_month_keys": int(df.duplicated(["canonical_project_id", "observation_month"]).sum()),
        "raw_date_nonmissing_before_conversion": raw_date_nonmissing,
        "date_columns_absent": [c for c in ("approval_date", "original_completion_date", "revised_completion_date") if c not in source_columns],
        "schedule_slippage_nonmissing": int(df["schedule_slippage_months"].notna().sum()),
        "historical_financial_scoring_available_rows": int(df["historical_financial_scoring_available"].sum()),
        "risk_distribution": df["risk_level"].value_counts(dropna=False).to_dict(),
        "latest_priority_distribution": latest["priority_level"].value_counts(dropna=False).to_dict(),
        "anomaly_flags": int(df["anomaly_flag"].sum()),
        "missingness": {c: int(v) for c, v in df.isna().sum().items()},
        "leakage_controls": ["sorted and differenced within project only", "no backward fill or zero imputation of source financial/progress/date fields", "month-local anomaly baseline", "latest revised-cost field excluded from historical scoring", "latest queue uses last observation per project"],
    }
    (OUT_DIR / "validation_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("input_shape", "unique_projects", "month_counts", "projects_with_all_seven_months", "duplicate_project_month_keys", "schedule_slippage_nonmissing", "risk_distribution", "latest_priority_distribution", "anomaly_flags")}, indent=2))
    print(f"Wrote: {features_path}\nWrote: {scores_path}\nWrote: {queue_path}")


if __name__ == "__main__":
    main()
