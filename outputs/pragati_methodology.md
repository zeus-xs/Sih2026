# Project Pragati methodology

The prototype is an AI-assisted risk screening and early-warning system. It does not claim to predict confirmed future project failure because validated future outcome labels are not available in the current prototype dataset.

It consumes the frozen feature dataset, selects each project's latest observation, and keeps project-month trajectories separate. Cost (30%), schedule (30%), progress (25%), and expenditure/physical imbalance (15%) are explainable dimensions. Dimensions without evidence are excluded and remaining weights are renormalized. Schedule is unavailable in the supplied data and is never treated as adverse evidence.

Scores use robust snapshot quantiles rather than unreviewed fixed thresholds. Categories are portfolio-calibrated: the upper 5% of sufficiently evidenced current projects are Critical, the next 15% High, and the next 30% Medium. Low-confidence projects become Review Data rather than high risk.

Data confidence reflects observed history, progress, finance, trajectory, dates, and identity evidence. Trajectories describe reported movement only: Improving, Stable, Stagnating, Deteriorating, or Insufficient Data.

The intended anomaly component is Isolation Forest on current project snapshots. The execution metadata records the actual backend. When scikit-learn is unavailable, a deterministic median/MAD robust outlier fallback is used and is not described as Isolation Forest. Anomaly means unusual portfolio behaviour, not confirmed delay.

Early warnings are traceable rules for adverse movement, divergence, escalation, anomalies, and data quality. Recommendations request monitoring or validation actions, not causal conclusions.

For production, retain dated cost revisions, completion dates, project-age fields, and outcome labels; then train and independently validate time-split supervised models. Do not report accuracy, precision, recall, F1, or AUC until such labels exist.
