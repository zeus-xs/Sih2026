from ml_common import build_snapshot, add_confidence, add_trajectory, add_domain_risk, OUT
def main():
 s=add_domain_risk(add_trajectory(add_confidence(build_snapshot()))); s.to_csv(OUT/'pragati_domain_risk_snapshot.csv',index=False); print('Wrote domain risk snapshot')
if __name__=='__main__': main()
