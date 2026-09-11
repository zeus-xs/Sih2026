from ml_common import warnings, OUT
def main():
 import pandas as pd
 s=pd.read_csv(OUT/'pragati_project_risk_profiles.csv'); w=warnings(s); w.to_csv(OUT/'pragati_early_warnings.csv',index=False); print(f'Wrote {len(w)} warnings')
if __name__=='__main__': main()
