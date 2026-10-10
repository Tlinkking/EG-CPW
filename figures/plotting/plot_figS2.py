"""Figure S2 (supplementary): S0-only false-positive BEV examples.

Companion to Figure 4: three S0-only FP cases (seed 3407, thr = 0.10)
from threshold_control_analysis/tables/qualitative_case_index.md:

  1. category 3: bicycle  token=c2848cc5... score 0.958  (background:
     no same-class GT in frame)
  2. category 3: barrier  token=90c191b4... score 0.914  (background)
  3. category 4: construction_vehicle token=7e3b2e21... score 0.900,
     nearest same-class GT at 2.99 m (near-target)

The highlighted magenta ring marks the S0-only FP box.

Run:  python plot_figS2.py
Out:  supplementary_figures/figS2_qualitative_fp.{pdf,png}
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bev_case import (C_C0, C_GT, C_HL, C_S0, SCORE_THR,  # noqa: E402
                      draw_case)
from figstyle import save  # noqa: E402

CASES = [
    dict(token='c2848cc504144003a6e1eac71a6918b9', xy=(-5.94, 5.26),
         half=10.0, score=0.958,
         title='bicycle, score = 0.958\nno same-class GT in frame'),
    dict(token='90c191b41e6641c9bd4a66b3cb90bf74', xy=(-6.97, -5.47),
         half=10.0, score=0.914,
         title='barrier, score = 0.914\nno same-class GT in frame'),
    dict(token='7e3b2e211788438c88a1840f5d2e6410', xy=(12.14, 0.28),
         half=12.0, score=0.900,
         title='construction_vehicle, score = 0.900\n'
               'nearest same-class GT at 2.99 m'),
]

fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.1))

for ax, case in zip(axes, CASES):
    draw_case(ax, case['token'], case['xy'][0], case['xy'][1],
              case['half'], fp_xy=case['xy'], fp_score=case['score'])
    ax.set_title(f'S0-only FP: {case["title"]}', fontsize=8)

legend = [Line2D([0], [0], color=C_GT, lw=2, label='GT box'),
          Line2D([0], [0], color=C_C0, lw=1.6, ls='--',
                 label=f'C0 pred (score ≥ {SCORE_THR})'),
          Line2D([0], [0], color=C_S0, lw=1.6,
                 label=f'S0 pred (score ≥ {SCORE_THR})'),
          Line2D([0], [0], marker='o', color=C_HL, ls='none',
                 markersize=8, markerfacecolor='none', markeredgewidth=1.6,
                 label='S0-only FP'),
          Line2D([0], [0], marker='.', color='0.55', ls='none',
                 markersize=6, label='LiDAR points')]
fig.legend(handles=legend, loc='lower center', ncol=5, fontsize=8,
           frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.suptitle('Qualitative examples of S0-only false positives '
             '(epoch-20, seed 3407, thr = 0.10)', fontsize=9.5)
fig.tight_layout(rect=[0, 0.105, 1, 0.95])
save(fig, 'figS2_qualitative_fp', supplementary=True)
