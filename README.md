# Project Pragati — SIH 2026 Problem 26013

This is a reproducible prototype for the current-state project risk scoring and early warning over the supplied Jan–Jul 2026 monitoring extract. It is **not** presented as a supervised predictor of future failure: the supplied data has seven monthly observations but no defensible future-risk outcome label.

## Run

From this workspace's project root:

```powershell
& 'C:\Users\Lenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' src\featureextract.py
```

On the user's Mac after copying this project into `/Users/anushajain/sih` and
installing the two dependencies in `requirements.txt`:

```bash
cd /Users/anushajain/sih
python3 -m pip install -r requirements.txt
python3 src/featureextract.py
```

The pipeline locates `data/pragati_jan_jul_2026_FINAL_CLEAN.csv` when supplied. Otherwise it uses the uploaded master CSV placed in `data/`. The input is never modified.

## Method

The pipeline calculates financial features (`cost_escalation_pct`, `expenditure_ratio_pct`, and `expenditure_progress_gap`), project-only monthly changes and 3/6-month velocities, schedule slippage when both source completion dates exist, and project age when an approval date exists. It keeps `source_file` and `source_page` when those are supplied.

The transparent risk dimensions are Cost 30%, Schedule 30%, Progress 25%, and Expenditure/Progress Imbalance 15%. Missing source measurements are never changed to zero. A dimension is excluded when its evidence is unavailable and the remaining available weights are renormalized; `risk_confidence` states how much of the intended weight was supported.

The uploaded master CSV labels revised cost as `latest_revised_cost_cr`, with no effective date. That can leak July information into earlier months, so the pipeline retains derived financial features for inspection but excludes Cost and Imbalance from historical scores. It similarly creates schedule and age fields as missing when their source columns are absent. This is intentional, auditable behaviour—not a synthetic substitute.

The AI layer is an unsupervised robust outlier detector. It calculates median/MAD deviations separately within each observation month, so it does not use later months as a baseline. Missing values are median-imputed only inside this model calculation; raw data remains missing. An anomaly flag is a review signal, not proof of wrongdoing and not a replacement for the transparent score.

## Outputs

- `outputs/pragati_features.csv` — source rows plus leakage-aware features.
- `outputs/pragati_risk_scores.csv` — one score and explanation per project-month.
- `outputs/pragati_priority_queue.csv` — one latest observation per project, ranked for review.
- `outputs/validation_report.json` — row/key/month/date/missingness/leakage checks.

## Limitations

No accuracy, precision, recall, F1, AUC, or future-prediction claim is made. The supplied CSV must contain observation-time revised-cost dates and completion dates before Cost, Imbalance, and Schedule can contribute fully. The source report filenames should also be reconciled with their observation months before using them to enrich the data.

## Judge-facing summary

Project Pragati gives reviewers an explainable, provenance-preserving priority queue, trend-based early-warning reasons, evidence confidence, and a separate anomaly signal. It does not conceal unavailable schedule or time-valid financial data; that makes it suitable for a monitored prototype and highlights the exact data enrichment needed for a production model.
