#!/usr/bin/env bash
# One-shot reproduction of the revised figure set (Figure 2-7, S1, S2).
# Reads existing dumps/tables only; no training, no inference.
set -euo pipefail

ROOT=/home/bjtc/projects_ws/mmdetection3d-1.3.0
SRC=$ROOT/work_dirs/evidence_controlled/paper_figures_epoch20/scripts

source /home/bjtc/programs/miniconda3/etc/profile.d/conda.sh
conda activate mmdet3d
export CUDA_VISIBLE_DEVICES=""

# 1) data export (skip if data/*.npz already exist; the script asserts
#    consistency with the round-1 exports)
if [ ! -f "$SRC/../data/sweep_matched_seed3408.npz" ]; then
  python "$SRC/export_figure_data.py"      # round-1 exports
  python "$SRC/export_figure_data_r2.py"   # round-2 exports
fi

# 2) figures
cd "$SRC/figures"
python plot_fig2.py    # figures/final/fig2_evidence_group_recall
python plot_fig3.py    # figures/final/fig3_distance_recall_delta
python plot_fig4.py    # figures/final/fig4_qualitative_bev
python plot_fig5.py    # supplementary_figures/figS1_threshold_sweep
python plot_fig6.py    # figures/final/fig6_evidence_weight_mechanism
python plot_fig7.py    # figures/final/fig7_gained_tp_vs_fp
python plot_figS2.py   # supplementary_figures/figS2_qualitative_fp

echo "All figures written to figures/final/ and supplementary_figures/"
