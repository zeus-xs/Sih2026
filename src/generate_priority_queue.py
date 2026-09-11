from ml_common import OUT
def main():
 import pandas as pd
 s=pd.read_csv(OUT/'pragati_project_risk_profiles.csv'); action=s[s.risk_category.isin(['CRITICAL','HIGH'])].sort_values('final_risk_score',ascending=False); monitor=s[s.risk_category.isin(['MEDIUM','LOW'])].sort_values('final_risk_score',ascending=False); review=s[s.risk_category.eq('REVIEW_DATA')].sort_values('data_confidence_score'); q=pd.concat([action,monitor,review]); q['priority_band']=['ACTION_REQUIRED']*len(action)+['MONITOR']*len(monitor)+['REVIEW_DATA']*len(review); q['priority_rank']=range(1,len(q)+1); q.to_csv(OUT/'pragati_priority_queue.csv',index=False); print(f'Wrote {len(q)} priorities')
if __name__=='__main__': main()
