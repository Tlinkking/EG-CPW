# Table 2. Evidence Group Breakdown (epoch-20, nuScenes val)

Sources: `work_dirs/evidence_controlled/analysis/C0full_vs_S0full_audit_report.txt`
(StrictMatch, frame-cluster bootstrap 1000 reps, seed=3407).

Grouping: per-class evidence quantiles on the fixed validation GT table
(Low = bottom 25%, Mid = middle 50%, High = top 25%); identical assignment
for both methods. Matching: StrictMatch (independent greedy one-to-one
assignment per threshold, same class, score ≥ 0.1).

| Group | GT count | Method | StrictMatch@0.5 | StrictMatch@1.0 | StrictMatch@2.0 (Recall) | SCMR_cond@0.5 | mCE@2m |
|---|---|---|---|---|---|---|---|
| Low | 47,449 | C0 | 0.2459 | 0.3196 | 0.3691 | 0.6016 | 0.5714 |
| Low | 47,449 | S0 | 0.2470 | 0.3252 | 0.3756 | 0.5906 | 0.5821 |
| Mid | 93,191 | C0 | 0.6638 | 0.7576 | 0.7967 | 0.8057 | 0.3495 |
| Mid | 93,191 | S0 | 0.6631 | 0.7586 | 0.7999 | 0.8002 | 0.3555 |
| High | 46,888 | C0 | 0.9238 | 0.9580 | 0.9718 | 0.9397 | 0.1963 |
| High | 46,888 | S0 | 0.9195 | 0.9596 | 0.9729 | 0.9342 | 0.1994 |

## Delta (S0 − C0) with 95% frame-cluster bootstrap CI

| Group | ΔStrictMatch@0.5 | ΔStrictMatch@1.0 | ΔStrictMatch@2.0 | ΔSCMR_cond@0.5 | ΔmCE@2m |
|---|---|---|---|---|---|
| Low | +0.0011 [−0.0018,+0.0038] | +0.0056 [+0.0030,+0.0082] | +0.0065 [+0.0040,+0.0093] | −0.0108 [−0.0177,−0.0040] | +0.0106 [+0.0036,+0.0176] |
| Mid | −0.0007 [−0.0027,+0.0013] | +0.0009 [−0.0007,+0.0025] | +0.0032 [+0.0017,+0.0046] | −0.0055 [−0.0081,−0.0030] | +0.0060 [+0.0036,+0.0083] |
| High | −0.0042 [−0.0062,−0.0021] | +0.0016 [+0.0000,+0.0031] | +0.0011 [−0.0002,+0.0024] | −0.0054 [−0.0076,−0.0033] | +0.0031 [+0.0012,+0.0050] |

Low-evidence GT profile: num_lidar_pts mean/median = 0.2/0;
38,048 of 47,449 Low GTs have zero points; zero-point GTs never entered
weighted supervision (dataset filter_empty_gt, num_lidar_pts > 0).
