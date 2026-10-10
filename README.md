# EG-CPW: Evidence-Guided Center-Peak Weighting for LiDAR 3D Object Detection

Official code release accompanying the EG-CPW paper (CenterPoint /
nuScenes). EG-CPW assigns a per-instance weight to the center-peak
positive loss of CenterPoint based on an observation-evidence score
(enclosed point count and volume-normalized point density):

```
e_i = beta * z_count_i + (1 - beta) * z_density_i        (per-class z-scores)
w_i = clip(1 + lambda * clip(-e_i, 0, q_max), 1, omega_max)
```

with `lambda = 0.25`, `q_max = 2.0`, `omega_max = 1.5`. Only low-evidence
instances are up-weighted; no instance is down-weighted below 1. The
Gaussian heatmap targets, box regression losses, decoder, NMS, and
inference are unchanged from CenterPoint.

This repository is an **overlay** on top of MMDetection3D 1.3.0 — it
contains only the files we added or modified, plus configs, analysis
scripts, and the result tables reported in the paper.

## Repository layout

```
EG-CPW-release/
├── egcpw/                  # EG-CPW core implementation (overlay onto mmdet3d)
│   ├── models/dense_heads/centerpoint_head.py
│   └── datasets/transforms/{__init__.py, compute_num_points.py,
│                            evidence_transforms.py, formating.py}
├── configs/centerpoint/    # CenterPoint + nuScenes + EG-CPW configs
├── tools/analysis/         # offline diagnostic / matching / audit scripts
├── figures/                # plotting scripts (final PDFs ship with the paper)
│   ├── export_figure_data.py, export_figure_data_r2.py
│   └── plotting/           # plot_fig2..7, plot_figS2, style helpers
├── scripts/                # run_train.sh, run_figures.sh
├── results/                # small result tables (md / json)
└── docs/
    ├── CHANGELOG.md
    ├── figure_captions_updated.md
    └── paper/              # paper source will be uploaded upon acceptance
```

## Installation

1. Install MMDetection3D 1.3.0 from source (tested environment:
   Ubuntu, Python 3.10, PyTorch 2.8.0 + CUDA 12.8, mmcv 2.1.0,
   mmdet 3.2.0, mmengine 0.10.7):

   ```bash
   git clone https://github.com/open-mmlab/mmdetection3d.git -b v1.3.0
   cd mmdetection3d
   pip install -e .
   ```

2. Overlay the EG-CPW files (see `egcpw/README.md` for exactly which
   stock files each file adds or replaces):

   ```bash
   cp egcpw/models/dense_heads/centerpoint_head.py \
      mmdet3d/models/dense_heads/centerpoint_head.py
   cp egcpw/datasets/transforms/*.py mmdet3d/datasets/transforms/
   cp configs/centerpoint/mycenterpoint_nus*.py configs/centerpoint/
   cp -r tools/analysis <mmdetection3d>/tools/
   ```

## Data preparation

Prepare nuScenes following the standard MMDetection3D instructions
(`tools/create_data.py nuscenes ...`) to produce
`nuscenes_infos_train.pkl` / `nuscenes_infos_val.pkl`. Then edit
`data_root` in the configs — the shipped configs contain the authors'
local path (`/home/bjtc/data/datasets/nuScenes/`) and must be pointed to
your own dataset location.

## Training

```bash
# S0: EG-CPW (peak_weight mode), 2 GPUs, 20 epochs
bash scripts/run_train.sh

# C0 baseline (evidence weighting disabled)
bash tools/dist_train.sh configs/centerpoint/mycenterpoint_nus_evidence_controlled.py 2 \
  --work-dir work_dirs/egcpw_baseline \
  --cfg-options model.train_cfg.pts.evidence_radius.enabled=False
```

Key config block (`model.train_cfg.pts.evidence_radius`):

| field | meaning | paper value |
|---|---|---|
| `enabled` | master switch | True (S0) / False (C0) |
| `mode` | `radius` (dynamic radius) or `peak_weight` (center-peak weighting) | `peak_weight` |
| `peak_weight_lambda` | weighting strength lambda | 0.25 |
| `peak_weight_max` | weight upper bound omega_max | 1.5 |
| `sparse_score_max` | deficit clip q_max | 2.0 |
| `beta` | evidence combination coefficient | 0.5 |
| `mu_count` / `std_count` / `mu_density` / `std_density` | per-class normalization statistics estimated once from the training split | (in config) |

## Evaluation and offline analysis

Standard evaluation uses the MMDetection3D test pipeline:

```bash
bash tools/dist_test.sh configs/centerpoint/mycenterpoint_nus_evidence_controlled.py \
  work_dirs/<run>/epoch_20.pth 2
```

The offline diagnostic analyses reported in the paper (StrictMatch@2.0 m,
evidence-group breakdown, threshold sweep, FP analysis) are under
`tools/analysis/`; figure reproduction is driven by `scripts/run_figures.sh`
(see `figures/README.md` for the two path constants to adjust).

## Results

The reported result tables (two seeds: 3407, 3408) are in `results/`
and the figure captions in `docs/figure_captions_updated.md`.

## Citation

If you find this code useful, please cite our paper (citation
information will be added upon publication).

## License

MIT (see LICENSE).
