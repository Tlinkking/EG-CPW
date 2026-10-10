# figures/ — plotting scripts

This directory contains only the plotting/export scripts. The final
paper PDFs/PNGs are **not** hosted here; they are submitted together
with the paper's LaTeX source.

- `export_figure_data.py`, `export_figure_data_r2.py` — read-only
  exporters that turn saved prediction dumps (`C0full_dump.pkl` /
  `S0full_dump.pkl`, produced by `tools/analysis/evidence_stratified_dump.py`
  from trained checkpoints) into the NPZ files under a local `data/`
  directory. Both scripts assert that re-computed quantities match the
  earlier audit exports before overwriting.
- `plotting/figstyle.py` — shared style (colorblind-safe Okabe-Ito
  palette, PDF + 600 dpi PNG output).
- `plotting/bev_case.py` — BEV case renderer used by the qualitative
  figures.
- `plotting/plot_fig2.py` ... `plot_fig7.py`, `plot_figS2.py` — one
  script per paper figure.

## Before running

A few scripts contain the authors' local absolute paths; adjust them to
your environment:

- `plotting/bev_case.py`: `ROOT` (mmdetection3d checkout), `INFOS`,
  `LIDAR_DIR` (nuScenes val infos and raw `LIDAR_TOP` sweeps).
- `../scripts/run_figures.sh`: `ROOT` and the conda activation path.

The figure scripts read their numeric inputs from the exported NPZ
files (Figures 3, 6, 7, S1c), from parsed result tables (Figure S1a/b),
or from values verified against the audit reports (Figure 2).

Run everything (from the mmdetection3d tree):

```bash
bash scripts/run_figures.sh
```
