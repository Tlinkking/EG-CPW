# Table 1. Overall Performance Comparison (epoch-20, nuScenes val)

Sources: `work_dirs/evidence_controlled/analysis/C0full_vs_S0full_audit_report.txt`,
`work_dirs/evidence_controlled/analysis/recall_fp_cost_analysis_epoch20.txt`,
training logs (`C0-full_baseline_20e_seed3407`, `S0-full_peak_weight_l0.25_20e_seed3407`).

## Detection statistics at protocol point (score threshold = 0.1, StrictMatch@2)

| Method | Recall | Precision | StrictMatch@0.5 | StrictMatch@1.0 | StrictMatch@2.0 | mCE@2m | Median CE@2m | p75 CE | p90 CE |
|---|---|---|---|---|---|---|---|---|---|
| C0 (baseline) | 0.7323 | 0.1824 | 0.6231 | 0.6969 | 0.7323 | 0.3270 | 0.1939 | 0.3726 | 0.7887 |
| S0 (peak weight λ=0.25) | 0.7358 | 0.1706 | 0.6219 | 0.6992 | 0.7358 | 0.3332 | 0.1963 | 0.3814 | 0.8131 |
| Δ (S0−C0) | +0.0035 | −0.0118 | −0.0011 | +0.0023 | +0.0035 | +0.0062 | +0.0024 | +0.0088 | +0.0244 |

Note: Recall is identical to StrictMatch@2.0 (GT-conditioned, matched under
independent 2.0 m one-to-one assignment). Precision = TP / predictions(score≥0.1).
Localization statistics are computed over StrictMatch@2 matched pairs.

## Official nuScenes metrics (epoch-20 checkpoint, standard evaluator)

| Method | mAP | NDS | mATE↓ | mASE↓ | mAOE↓ | mAVE↓ | mAAE↓ |
|---|---|---|---|---|---|---|---|
| C0 (baseline) | 0.5644 | 0.6455 | 0.3047 | 0.2550 | 0.3321 | 0.2834 | 0.1918 |
| S0 (peak weight λ=0.25) | 0.5655 | 0.6474 | 0.3083 | 0.2534 | 0.3061 | 0.2955 | 0.1902 |
| Δ (S0−C0) | +0.0011 | +0.0019 | +0.0036 | −0.0016 | −0.0260 | +0.0121 | −0.0016 |
