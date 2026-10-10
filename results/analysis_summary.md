# Table 4. Failure Analysis Statistics (epoch-20, score threshold = 0.1)

Sources: `work_dirs/evidence_controlled/analysis/proposal_nms_analysis_epoch20.txt`,
`work_dirs/evidence_controlled/analysis/score_zero_analysis_epoch20.txt`,
`work_dirs/evidence_controlled/analysis/recall_fp_cost_analysis_epoch20.txt`.

S0-only prediction: S0 prediction (score ≥ 0.1) with no same-class C0
prediction within 2.0 m (greedy pred-to-pred matching, score descending).
S0-only predictions: 345,067 (TP gain 4,638; FP increase 340,429).

## 4a. S0-only FP distance to nearest same-class GT (thr = 0.1)

| Distance bucket | Count | Share of S0-only FP |
|---|---|---|
| 0–1 m | 12,844 | 0.0377 |
| 1–2 m | 8,470 | 0.0249 |
| 2–5 m | 18,402 | 0.0541 |
| 5–10 m | 22,572 | 0.0663 |
| >10 m | 122,347 | 0.3593 |
| background (no same-class GT in frame) | 155,794 | 0.4576 |
| total | 340,429 | 1.0000 |

## 4b. S0-only FP class distribution (thr = 0.1)

| Class | Count | Share |
|---|---|---|
| bicycle | 55,258 | 0.1623 |
| barrier | 49,912 | 0.1466 |
| pedestrian | 49,537 | 0.1455 |
| traffic_cone | 48,182 | 0.1415 |
| car | 41,792 | 0.1228 |
| motorcycle | 34,974 | 0.1027 |
| truck | 23,079 | 0.0678 |
| construction_vehicle | 19,841 | 0.0583 |
| trailer | 13,204 | 0.0388 |
| bus | 4,650 | 0.0137 |

## 4c. TP / FP score distribution (StrictMatch@2, thr = 0.1)

| Method | TP count | FP count | TP mean | TP median | FP mean | FP median |
|---|---|---|---|---|---|---|
| C0 | 137,322 | 615,444 | 0.6023 | 0.6612 | 0.1863 | 0.1463 |
| S0 | 137,984 | 670,849 | 0.6028 | 0.6622 | 0.1858 | 0.1462 |
| Δ | +662 | +55,405 | +0.0005 | +0.0009 | −0.0005 | −0.0001 |

## 4d. TP score by evidence group of the matched GT

| Group | C0 TP mean | S0 TP mean | C0 TP median | S0 TP median |
|---|---|---|---|---|
| Low | 0.3651 | 0.3725 | 0.3377 | 0.3468 |
| Mid | 0.5699 | 0.5738 | 0.6165 | 0.6211 |
| High | 0.7463 | 0.7401 | 0.7868 | 0.7812 |
