# Updated Figure Captions (Figure 2–7 + Supplementary)

Scope: CenterPoint on nuScenes val, epoch-20 checkpoints, seeds 3407 and
3408, C0 (baseline) vs S0 (evidence-guided center-peak weighting,
λ = 0.25, ω_max = 1.5). Unless stated otherwise, matching is
StrictMatch@2.0 m (class-aware greedy one-to-one assignment) at score
threshold 0.10, and the evidence groups (Low / Mid / High) come from the
per-class 25th/75th percentile partition of the evidence score on the
fixed validation GT table (n = 187,528 GTs; group sizes 47,449 / 93,191 /
46,888). Evidence groups are an offline analysis stratification computed
from GT statistics; they are not used at inference time. All offline
metrics below are diagnostic analyses on saved prediction dumps, not
official nuScenes leaderboard metrics; official mAP/NDS are reported in
Table 1 (overall_performance.md).

File naming and in-text references:

| In-text | File (PDF + 600 dpi PNG) |
|---|---|
| Figure 2 | figures/final/fig2_evidence_group_recall.{pdf,png} |
| Figure 3 | figures/final/fig3_distance_recall_delta.{pdf,png} |
| Figure 4 | figures/final/fig4_qualitative_bev.{pdf,png} |
| Figure 6 | figures/final/fig6_evidence_weight_mechanism.{pdf,png} |
| Figure 7 | figures/final/fig7_gained_tp_vs_fp.{pdf,png} |
| Figure S1 | supplementary_figures/figS1_threshold_sweep.{pdf,png} |
| Figure S2 | supplementary_figures/figS2_qualitative_fp.{pdf,png} |

---

## Figure 2 — Evidence-group recall (absolute + delta)

**Suggested caption:**
Evidence-group recall on nuScenes val (epoch-20, seeds 3407/3408,
StrictMatch@2.0 m, score threshold 0.10). (a) Absolute recall of C0
(baseline) and S0 (center-peak weighting) per evidence group; group GT
counts are given under each group label. (b) Recall difference
(S0 − C0) with 95% confidence intervals from a per-seed frame-cluster
bootstrap (1000 resamples over frames). The Low-evidence group improves
by +0.65 pp (seed 3407) and +1.38 pp (seed 3408), the Mid group by
+0.32 pp and +0.55 pp, and the High-evidence group changes little
(≤ +0.11 pp; 95% CI includes zero in both seeds). Evidence groups are
an offline diagnostic stratification of the validation GT table.

*Data source:* C0full_vs_S0full_audit_report.txt (seed 3407 and seed
3408 analyses).

## Figure 3 — Recall vs ground-truth center distance

**Suggested caption:**
Diagnostic view of how the recall difference distributes over the
ground-truth center distance (5 m bins, ego-centered BEV distance,
StrictMatch@2.0 m, score threshold 0.10). Top: absolute recall of C0
and S0 for all GTs and per evidence group (solid: seed 3407, dashed:
seed 3408). Bottom: per-bin recall difference (S0 − C0) with paired
95% frame-cluster bootstrap confidence intervals (1000 resamples over
frames; filled markers denote bins whose CI excludes zero). Bin labels
in the Overall panel give the GT count per bin; bins with fewer than
30 GTs are omitted. The recall difference concentrates in the Low and
Mid evidence groups at near-to-mid ranges; high-evidence objects are
already recalled at > 0.95 by both methods at most distances. This is
an offline diagnostic analysis of where the differences occur along
the distance axis.

*Data source:* data/distance_recall_seed{3407,3408}.npz (exported from
the epoch-20 prediction dumps).

## Figure 4 — Qualitative BEV examples

**Suggested caption:**
Representative qualitative examples of low-evidence ground-truth
objects recalled by S0 but missed by C0 (epoch-20, seed 3407, score
threshold 0.10, StrictMatch@2.0 m). Green solid boxes: ground truth
(with heading); orange dashed boxes: C0 predictions (score ≥ 0.10);
blue solid boxes: S0 predictions; gray dots: raw LiDAR points; star +
dashed circle: the target low-evidence GT. Left: a traffic cone with 0
enclosed LiDAR points (evidence e = −2.687); middle: a pedestrian with
1 enclosed point (e = −2.046); right: a car with 0 enclosed points
(e = −1.674). For zero-point targets the input contains no object
points, so detection may leverage scene context and class priors.
Ground-truth boxes and evidence labels are used for offline annotation
and visualization only. Additional qualitative examples including
S0-only false positives are given in Supplementary Figure S2.

*Data source:* qualitative_case_index.md (category 1), epoch-20
prediction dumps, nuscenes_infos_val.pkl (GT heading), raw
LIDAR_TOP sweeps.

## Figure 5 → Supplementary Figure S1 — Score-threshold sweep

**Main-text sentence (replaces the figure in the main paper):**
"A score-threshold sweep shows no PR dominance between C0 and S0; the
operating curves nearly coincide (best F1 = 0.6557 vs 0.6559, both at
threshold 0.40). See Supplementary Figure S1."

**Suggested caption (Figure S1):**
Score-threshold sweep (0.10–0.80) on nuScenes val, epoch-20, seeds
3407/3408. (a) Overall PR operating curve (overall prediction-level
precision vs overall StrictMatch@2.0 recall). (b) Group-restricted
recall (Low/Mid/High evidence groups, group GT denominator) vs overall
precision at the same threshold; note that the precision axis is the
overall prediction-level precision, not a per-group precision.
(c) Per-threshold recall difference (S0 − C0) with 95% frame-cluster
bootstrap confidence intervals (1000 resamples over frames). The C0
and S0 operating curves nearly coincide across the sweep.

*Data source:* threshold_control_analysis/tables/evidence_group_curve.md
(panels a, b); data/sweep_matched_seed{3407,3408}.npz (panel c).

## Figure 6 — Evidence-weighting mechanism

**Suggested caption:**
Mechanism of the evidence-guided center-peak weighting (training-time,
S0). (a) Deterministic evidence-to-weight mapping
ω_i = clip(1 + λ·clip(−e_i, 0, q_max), 1, ω_max) with λ = 0.25,
q_max = 2.0, ω_max = 1.5; weights saturate at ω_max for e_i ≤ −2.0 and
never drop below 1. (b) Evidence-score distribution of the validation
GT table (n = 187,528) per evidence group (per-class 25th/75th
percentile partition; group boundaries differ by class). (c) Enclosed
point count vs volume-normalized point density (log–log, random
subsample n = 20,000), colored by the assigned weight; up-weighted
instances concentrate in the low-count/low-density region. (d)
Per-class median weight and share of GTs with ω_i > 1. The weighting
concentrates on exactly the low-evidence population whose recall
difference is shown in Figures 2 and 3.

*Data source:* data/evidence_scatter.npz (validation GT table from the
seed-3407 dump, identical for C0/S0); mapping parameters from the
training configuration.

## Figure 7 — Gained TP vs gained FP (offline diagnostic)

**Suggested caption:**
Offline diagnostic comparison of the predictions that change between
C0 and S0 at the protocol operating point (score threshold 0.10,
StrictMatch@2.0 m, epoch-20). Gained TPs are ground-truth objects
matched by S0 but not by C0 (n = 4,858 / 5,256 for seeds 3407 / 3408);
gained FPs are S0 predictions without a same-class C0 prediction within
2.0 m that remain unmatched to ground truth (n = 340,429 / 343,249).
(a) Score distributions of gained TPs and gained FPs (seed 3407).
(b) Distance to the nearest same-class GT center: matched distance for
gained TPs (≤ 2 m by construction) and CDF for gained FPs; the share
of gained FPs in frames without any same-class GT (background) is
45.8% (seed 3407) and 45.1% (seed 3408). (c) Class composition of the
two sets (seed 3407). (d) Evidence-group composition of gained TPs
(Low 41.1% / Mid 49.7% / High 9.2% at seed 3407) and the context split
of gained FPs, for both seeds. The additional false positives are
predominantly background-related, and the FP cost per unit recall gain
decreases sharply at higher score thresholds (from 156,948 additional
FPs per +0.01 recall at threshold 0.10 to 450 at threshold 0.60,
seed 3407).

*Data source:* data/gained_tp_seed{3407,3408}.npz,
data/fp_distance_seed{3407,3408}.npz, and
analysis/recall_fp_cost_analysis_epoch20.txt (FP-cost table).

## Supplementary Figure S2 — Qualitative S0-only FP examples

**Suggested caption:**
Qualitative examples of S0-only false positives (epoch-20, seed 3407,
score threshold 0.10; magenta rings). Left and middle: high-score
background false positives (bicycle, score 0.958; barrier, score 0.914)
in frames without any same-class ground truth. Right: a near-target
false positive (construction vehicle, score 0.900) 2.99 m from the
nearest same-class ground truth. Box/point rendering follows Figure 4.

*Data source:* qualitative_case_index.md (categories 3 and 4), epoch-20
prediction dumps, raw LIDAR_TOP sweeps.
