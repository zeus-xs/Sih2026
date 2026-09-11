"""Shared, leakage-aware intelligence layer for Project Pragati."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FEATURES = OUT / "pragati_features.csv"

# These columns were produced by the earlier feature-stage prototype.  They
# are not part of the canonical AI/ML handoff contract and must not appear
# beside the final AI-layer score/category in dashboard-facing outputs.
LEGACY_RISK_OUTPUT_COLUMNS = [
    "risk_score", "risk_level", "overall_risk_score", "priority_rank",
    "cost_risk", "schedule_risk", "progress_risk", "imbalance_risk",
    "risk_evidence_weight", "risk_confidence", "risk_reasons",
]

def read_features():
    d = pd.read_csv(FEATURES)
    d["observation_month"] = pd.to_datetime(d["observation_month"], errors="coerce")
    d["canonical_project_id"] = d["canonical_project_id"].astype("string")
    return d

def num(d, c): return pd.to_numeric(d[c], errors="coerce") if c in d else pd.Series(np.nan, index=d.index)

def qscale(s, lower=.50, upper=.95, positive=True):
    """Robust 0--100 scale; values below the median/default baseline are zero."""
    x = pd.to_numeric(s, errors="coerce")
    if positive: x = x.clip(lower=0)
    valid = x.dropna()
    if valid.empty: return pd.Series(np.nan, index=x.index)
    lo, hi = valid.quantile(lower), valid.quantile(upper)
    if not np.isfinite(hi) or hi <= lo: return pd.Series(np.where(x.notna() & (x > lo), 100.0, 0.0), index=x.index)
    return ((x - lo) / (hi - lo) * 100).clip(0, 100)

def build_snapshot(d=None):
    d = read_features() if d is None else d.copy()
    d = d.sort_values(["canonical_project_id", "observation_month"], kind="stable")
    history = d.groupby("canonical_project_id").agg(
        observed_months=("observation_month", "nunique"),
        first_observation_month=("observation_month", "min"),
        latest_observation_month=("observation_month", "max"),
    ).reset_index()
    history["history_length_months"] = history["observed_months"]
    history["complete_7_month_history"] = history["observed_months"].eq(7)
    latest = d.groupby("canonical_project_id", sort=False).tail(1).copy()
    latest = latest.merge(history, on="canonical_project_id", how="left", validate="one_to_one")
    return latest

def add_confidence(s):
    months = (s["observed_months"].clip(upper=7) / 7 * 30)
    progress = num(s,"physical_progress_pct").notna().astype(float) * 20
    finance = (num(s,"original_cost_cr").gt(0) & num(s,"revised_cost_cr").gt(0) & num(s,"cumulative_expenditure_cr").notna()).astype(float)*20
    trajectory = num(s,"progress_velocity_3m").notna().astype(float)*15
    dates = (num(s,"project_age_months").notna() | num(s,"schedule_slippage_months").notna()).astype(float)*10
    identity = s.get("match_status", pd.Series("",index=s.index)).eq("EXACT_MASTER_ID").astype(float)*5
    s["data_confidence_score"] = (months+progress+finance+trajectory+dates+identity).round(1)
    s["data_confidence_category"] = pd.cut(s["data_confidence_score"],[-1,39,59,79,100],labels=["INSUFFICIENT_EVIDENCE","LOW","MEDIUM","HIGH"]).astype("string")
    return s

def add_trajectory(s):
    v, n = num(s,"progress_velocity_3m"), s["observed_months"]
    upper = v.dropna().quantile(.75) if v.notna().any() else 1
    s["trajectory_status"] = np.select([n<3, v<0, v<=.25, v>=upper],["INSUFFICIENT_DATA","DETERIORATING","STAGNATING","IMPROVING"],default="STABLE")
    s["trajectory_risk_score"] = s["trajectory_status"].map({"DETERIORATING":85,"STAGNATING":60,"STABLE":30,"IMPROVING":10,"INSUFFICIENT_DATA":np.nan})
    return s

def add_domain_risk(s):
    # Latest snapshot only: no later value is ever written into a historical row.
    s["cost_risk_score"] = qscale(num(s,"cost_escalation_pct"), .50, .95)
    s["imbalance_risk_score"] = qscale(num(s,"expenditure_progress_gap"), .75, .95)
    velocity = num(s,"progress_velocity_3m").combine_first(num(s,"progress_change_1m"))
    ref = velocity.dropna(); lo, hi = (ref.quantile(.10), ref.quantile(.75)) if not ref.empty else (0,1)
    s["progress_risk_score"] = ((hi-velocity)/(hi-lo)*100).clip(0,100) if hi>lo else pd.Series(np.nan,index=s.index)
    s["schedule_data_available"] = num(s,"schedule_slippage_months").notna()
    s["schedule_risk_score"] = np.where(s["schedule_data_available"],qscale(num(s,"schedule_slippage_months"),.50,.95),np.nan)
    dims={"cost_risk_score":.30,"schedule_risk_score":.30,"progress_risk_score":.25,"imbalance_risk_score":.15}
    den=sum(s[c].notna()*w for c,w in dims.items()); nume=sum(s[c].fillna(0)*w for c,w in dims.items())
    s["base_risk_score"]=(nume/den.replace(0,np.nan)).round(2)
    s["risk_dimension_evidence_weight"]=den
    return s

def robust_anomaly(s):
    """Project-level anomaly detection using sklearn IsolationForest."""
    from sklearn.ensemble import IsolationForest
    import joblib

    requested = [
        "cost_escalation_pct",
        "expenditure_ratio_pct",
        "physical_progress_pct",
        "expenditure_progress_gap",
        "progress_velocity_3m",
        "expenditure_velocity_3m",
        "project_age_months",
    ]

    used = [
        c for c in requested
        if c in s and num(s, c).notna().sum() >= 20
    ]

    if not used:
        raise RuntimeError(
            "IsolationForest has no usable numerical features."
        )

    x = pd.DataFrame({c: num(s, c) for c in used})

    # Median imputation on the current project snapshot.
    medians = x.median()
    x = x.fillna(medians)

    if x.isna().any().any():
        raise RuntimeError(
            "IsolationForest input still contains missing values."
        )

    model = IsolationForest(
        n_estimators=300,
        contamination=0.05,
        random_state=42,
        n_jobs=-1,
    )

    # ACTUAL IsolationForest training.
    model.fit(x)

    # Higher anomaly_score = more unusual.
    s["anomaly_score"] = -model.decision_function(x)

    # -1 = anomaly, +1 = normal.
    s["anomaly_flag"] = model.predict(x) == -1

    p95 = s["anomaly_score"].quantile(0.95)
    p99 = s["anomaly_score"].quantile(0.99)

    s["anomaly_category"] = np.select(
        [
            s["anomaly_score"] >= p99,
            s["anomaly_score"] >= p95,
        ],
        [
            "HIGHLY_UNUSUAL",
            "UNUSUAL",
        ],
        default="NORMAL",
    )

    model_path = (
        ROOT / "models" / "pragati_isolation_forest.joblib"
    )
    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "model": model,
            "features": used,
            "medians": medians.to_dict(),
        },
        model_path,
    )

    metadata = {
        "backend": "sklearn_isolation_forest",
        "model_type": "IsolationForest",
        "implementation": (
            "sklearn.ensemble.IsolationForest"
        ),
        "fit_executed": True,
        "contamination": 0.05,
        "random_state": 42,
        "n_estimators": 300,
        "observations": int(len(s)),
        "features_used": used,
        "number_of_projects_used": int(len(s)),
        "number_of_anomalies": int(
            s["anomaly_flag"].sum()
        ),
        "anomaly_rate": round(
            float(s["anomaly_flag"].mean()),
            6,
        ),
        "anomaly_rate_percent": round(
            float(s["anomaly_flag"].mean() * 100),
            2,
        ),
        "model_path": str(model_path),
    }

    (
        OUT / "anomaly_model_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    return s, metadata

def fuse(s):
    """
    Fuse current-state risk with forward-looking predictive risk.

    Components:
        70% current domain risk
        15% trajectory risk
        10% anomaly signal
         5% XGBoost predictive deterioration risk

    XGBoost is intentionally given a bounded contribution because
    its forward-looking validation performance is still modest.
    """

    s = s.copy()

    # ---------------------------------------------------------
    # 1. ANOMALY COMPONENT
    # ---------------------------------------------------------

    anomaly_component = qscale(
        s["anomaly_score"],
        .50,
        .95,
    )

    s["anomaly_risk_component"] = (
        anomaly_component
    )

    # ---------------------------------------------------------
    # 2. TRAJECTORY COMPONENT
    # ---------------------------------------------------------

    trajectory_component = (
        s["trajectory_risk_score"]
        .fillna(0)
    )

    s["trajectory_risk_component"] = (
        trajectory_component
    )

    # ---------------------------------------------------------
    # 3. PREDICTIVE COMPONENT
    # ---------------------------------------------------------

    if "future_deterioration_probability" in s.columns:

        predictive_probability = pd.to_numeric(
            s[
                "future_deterioration_probability"
            ],
            errors="coerce",
        )

        s["predictive_risk_score"] = (
            predictive_probability
            .clip(0, 1)
            * 100
        ).round(2)

        predictive_component = (
            s["predictive_risk_score"]
            .fillna(0)
        )

        predictive_available = (
            s[
                "future_deterioration_probability"
            ].notna()
        )

    else:

        s["predictive_risk_score"] = np.nan

        predictive_component = pd.Series(
            0.0,
            index=s.index,
        )

        predictive_available = pd.Series(
            False,
            index=s.index,
        )

    # ---------------------------------------------------------
    # 4. CURRENT DOMAIN RISK
    # ---------------------------------------------------------

    current_component = pd.to_numeric(
        s["base_risk_score"],
        errors="coerce",
    )

    # ---------------------------------------------------------
    # 5. WEIGHTED FUSION
    # ---------------------------------------------------------

    # Full model:
    #
    # 70% current domain risk
    # 15% trajectory
    # 10% anomaly
    #  5% predictive deterioration
    #
    # If predictive data is unavailable, its weight is
    # redistributed proportionally across the available
    # current-state components.

    weights = {
        "current": 0.70,
        "trajectory": 0.15,
        "anomaly": 0.10,
        "predictive": 0.05,
    }

    available_current = (
        current_component.notna()
    )

    available_trajectory = (
        s["trajectory_risk_score"].notna()
    )

    available_anomaly = (
        s["anomaly_score"].notna()
    )

    available_predictive = (
        predictive_available
    )

    weighted_sum = (
        current_component.fillna(0)
        * weights["current"]
    )

    denominator = (
        available_current.astype(float)
        * weights["current"]
    )

    weighted_sum += (
        trajectory_component
        * weights["trajectory"]
    )

    denominator += (
        available_trajectory.astype(float)
        * weights["trajectory"]
    )

    weighted_sum += (
        anomaly_component.fillna(0)
        * weights["anomaly"]
    )

    denominator += (
        available_anomaly.astype(float)
        * weights["anomaly"]
    )

    weighted_sum += (
        predictive_component
        * weights["predictive"]
    )

    denominator += (
        available_predictive.astype(float)
        * weights["predictive"]
    )

    s["final_risk_score"] = (
        weighted_sum
        / denominator.replace(0, np.nan)
    ).round(2)

    # ---------------------------------------------------------
    # 6. DATA SUFFICIENCY
    # ---------------------------------------------------------

    insufficient = (
        (s["data_confidence_score"] < 40)
        | s["base_risk_score"].isna()
        | s["final_risk_score"].isna()
    )

    # ---------------------------------------------------------
    # 7. RISK CATEGORIES
    # ---------------------------------------------------------

    valid_scores = s.loc[
        ~insufficient,
        "final_risk_score",
    ]

    if not valid_scores.empty:

        q50 = valid_scores.quantile(.50)
        q80 = valid_scores.quantile(.80)
        q95 = valid_scores.quantile(.95)

    else:

        q50 = 50
        q80 = 70
        q95 = 85

    s["risk_category"] = np.select(
        [
            insufficient,
            s["final_risk_score"] >= q95,
            s["final_risk_score"] >= q80,
            s["final_risk_score"] >= q50,
        ],
        [
            "REVIEW_DATA",
            "CRITICAL",
            "HIGH",
            "MEDIUM",
        ],
        default="LOW",
    )

    # ---------------------------------------------------------
    # 8. RECORD THE FUSION METHODOLOGY
    # ---------------------------------------------------------

    s["risk_fusion_method"] = (
        "70% current domain + "
        "15% trajectory + "
        "10% anomaly + "
        "5% predictive deterioration"
    )

    return s

def add_explanations(s):
    def drivers(r):
        candidates=[]
        if pd.notna(r.cost_risk_score) and r.cost_risk_score>=70: candidates.append((r.cost_risk_score,"HIGH_COST_ESCALATION"))
        if pd.notna(r.imbalance_risk_score) and r.imbalance_risk_score>=70: candidates.append((r.imbalance_risk_score,"HIGH_EXPENDITURE_PROGRESS_GAP"))
        if r.trajectory_status=="DETERIORATING": candidates.append((85,"PROGRESS_DETERIORATION"))
        elif r.trajectory_status=="STAGNATING": candidates.append((60,"PROGRESS_STAGNATION"))
        if r.anomaly_flag: candidates.append((75 if r.anomaly_category=="HIGHLY_UNUSUAL" else 55,"ANOMALOUS_PROJECT_PATTERN"))
        if r.data_confidence_score<40: candidates.append((40,"INSUFFICIENT_DATA"))
        labels=[x[1] for x in sorted(candidates,reverse=True)] or ["LIMITED_ADVERSE_EVIDENCE"]
        return labels[:3]
    values=s.apply(drivers,axis=1)
    for i in range(3): s[f"risk_driver_{i+1}"]=values.map(lambda x:x[i] if len(x)>i else pd.NA)
    s["primary_risk_driver"]=s.risk_driver_1; s["secondary_risk_driver"]=s.risk_driver_2
    s["risk_explanation"]=s.apply(lambda r: "Data review is required before risk escalation." if r.risk_category=="REVIEW_DATA" else "; ".join([str(x).replace("_"," ").title() for x in [r.risk_driver_1,r.risk_driver_2] if pd.notna(x)]),axis=1)
    s["recommended_action"]=np.select([s.risk_category.isin(["CRITICAL","HIGH"]),s.risk_category.eq("REVIEW_DATA"),s.trajectory_status.eq("STAGNATING")],["Prioritise milestone-level review and identify implementation bottlenecks.","Verify missing/ambiguous monitoring data before escalation.","Monitor next reporting cycle and request a milestone update."],default="Continue routine portfolio monitoring.")
    return s

def warnings(s):
    rows=[]
    for _,r in s.iterrows():
        rules=[]
        if r.trajectory_status=="STAGNATING": rules.append(("PROGRESS_STAGNATION","HIGH","progress_velocity_3m",r.progress_velocity_3m,"Physical progress has shown minimal recent movement.","Prioritise milestone-level review and identify implementation bottlenecks."))
        if r.trajectory_status=="DETERIORATING": rules.append(("PROGRESS_DETERIORATION","CRITICAL","progress_velocity_3m",r.progress_velocity_3m,"Recent physical progress is declining.","Escalate for a milestone and data-quality review."))
        if pd.notna(r.imbalance_risk_score) and r.imbalance_risk_score>=70: rules.append(("EXPENDITURE_PROGRESS_DIVERGENCE","HIGH","expenditure_progress_gap",r.expenditure_progress_gap,"Expenditure is unusually ahead of physical progress relative to the portfolio.","Review expenditure milestones against physical deliverables."))
        if pd.notna(r.cost_risk_score) and r.cost_risk_score>=70: rules.append(("COST_ESCALATION","HIGH","cost_escalation_pct",r.cost_escalation_pct,"Cost escalation is high relative to the current portfolio.","Review revised-cost basis and mitigation plan."))
        if r.anomaly_flag: rules.append(("ANOMALOUS_TRAJECTORY","HIGH" if r.anomaly_category=="HIGHLY_UNUSUAL" else "MEDIUM","anomaly_score",r.anomaly_score,"Observed pattern is unusual relative to the current portfolio.","Review the underlying reporting and project milestones."))
        if r.data_confidence_score<40: rules.append(("DATA_QUALITY_REVIEW","LOW","data_confidence_score",r.data_confidence_score,"Evidence is insufficient for a dependable risk assessment.","Validate project identity and missing monitoring fields."))
        for typ,sev,key,val,desc,act in rules: rows.append({"warning_id":f"{r.canonical_project_id}-{typ}","project_id":r.project_id,"canonical_project_id":r.canonical_project_id,"warning_type":typ,"severity":sev,"trigger_value":val,"trigger_description":desc,"recommended_action":act})
    return pd.DataFrame(rows)
