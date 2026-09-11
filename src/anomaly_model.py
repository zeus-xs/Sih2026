from ml_common import build_snapshot, add_confidence, add_trajectory, add_domain_risk, robust_anomaly, OUT
def main():
 s,_=robust_anomaly(add_domain_risk(add_trajectory(add_confidence(build_snapshot())))); s.to_csv(OUT/'pragati_anomaly_snapshot.csv',index=False); print('Wrote anomaly snapshot and metadata')
if __name__=='__main__': main()
