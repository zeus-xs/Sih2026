from ml_common import build_snapshot, OUT
def main():
 s=build_snapshot(); s.to_csv(OUT/'pragati_project_snapshot.csv',index=False); print(f'Wrote {len(s)} project snapshots')
if __name__=='__main__': main()
