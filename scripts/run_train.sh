#!/usr/bin/env bash
# EG-CPW training example (CenterPoint, nuScenes, 2 GPUs, 20 epochs).
# Run from the root of your local MMDetection3D 1.3.0 tree, after
# overlaying the egcpw/ files and copying configs/ into configs/centerpoint/.
set -euo pipefail

GPUS=${GPUS:-2}
CFG=configs/centerpoint/mycenterpoint_nus_evidence_controlled.py
WORK_DIR=${WORK_DIR:-work_dirs/egcpw_peak_weight}

# S0: evidence-guided center-peak weighting (peak_weight mode)
bash tools/dist_train.sh "$CFG" "$GPUS" \
  --work-dir "$WORK_DIR" \
  --cfg-options model.train_cfg.pts.evidence_radius.mode=peak_weight

# C0 baseline: disable evidence weighting
# bash tools/dist_train.sh "$CFG" "$GPUS" \
#   --work-dir work_dirs/egcpw_baseline \
#   --cfg-options model.train_cfg.pts.evidence_radius.enabled=False
