# Changelog

## v1.0.0 — Initial public release (2026-10)

Code release accompanying the EG-CPW paper.

- EG-CPW core implementation as an overlay on MMDetection3D 1.3.0
  (`egcpw/`): evidence score, `radius` and `peak_weight` modes in
  `CenterHead`, point-count transforms.
- Configs: `mycenterpoint_nus.py`, `mycenterpoint_nus_evidence_controlled.py`,
  `mycenterpoint_nus_peak_weight.py` (CenterPoint, nuScenes, 20 epochs,
  voxel 0.075 / pillar 0.2, seeds 3407/3408 as reported).
- Offline analysis scripts (`tools/analysis/`): dump export,
  StrictMatch matching audit, evidence-stratified analysis, mechanism
  (weight-mass) analysis, score/zero-point analysis, proposal/NMS
  analysis, recall–FP-cost analysis.
- Figure pipeline (`figures/`): data exporters + per-figure plotting
  scripts (Figures 2–7, S1, S2). Final figure PDFs ship with the paper
  LaTeX package, not in this repository.
- Result tables (`results/`): overall performance, evidence-group
  breakdown, threshold operating points, analysis summary, and the
  threshold-control results JSON (two seeds).
- Docs: figure captions, this changelog, and `docs/paper/` placeholder.

Internal audit notes and intermediate analysis logs are intentionally
not part of the public release.
