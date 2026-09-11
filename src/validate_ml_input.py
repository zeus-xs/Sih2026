from __future__ import annotations
import json, numpy as np
from ml_common import read_features, OUT, num
def main():
 d=read_features(); required=["canonical_project_id","observation_month","physical_progress_pct","cumulative_expenditure_cr"]
 report={"shape":list(d.shape),"required_columns_present":{c:c in d for c in required},"unique_projects":int(d.canonical_project_id.nunique()),"month_count":int(d.observation_month.nunique()),"duplicate_project_month_keys":int(d.duplicated(["canonical_project_id","observation_month"]).sum()),"complete_history_projects":int((d.groupby("canonical_project_id").observation_month.nunique()==7).sum()),"infinite_numeric_values":int(np.isinf(d.select_dtypes(include="number")).sum().sum()),"missingness":{c:int(v) for c,v in d.isna().sum().items()},"range_flags":{"physical_progress_outside_0_100":int(((num(d,"physical_progress_pct")<0)|(num(d,"physical_progress_pct")>100)).sum()),"extreme_cost_escalation_abs_gt_500":int((num(d,"cost_escalation_pct").abs()>500).sum()),"schedule_available_rows":int(num(d,"schedule_slippage_months").notna().sum())}}
 (OUT/"ml_input_validation.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps({k:report[k] for k in ['shape','unique_projects','month_count','duplicate_project_month_keys','complete_history_projects','infinite_numeric_values']},indent=2))
if __name__=="__main__": main()
