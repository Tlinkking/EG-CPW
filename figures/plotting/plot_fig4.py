"""Figure 4 (main): qualitative BEV examples of low-evidence GTs
recalled by S0.

Three representative category-1 cases from
threshold_control_analysis/tables/qualitative_case_index.md
(seed 3407; Low-evidence GT, C0 unmatched / S0 matched,
thr = 0.10, StrictMatch@2.0):

  1. traffic_cone  token=1fa5506c...  gt_idx=37  0 pts  e=-2.687
  2. pedestrian    token=1477ef1a...  gt_idx=2   1 pt   e=-2.046
  3. car           token=3218ed6b...  gt_idx=24  0 pts  e=-1.674

Zero-point GTs are explicitly annotated on the panel.
GT boxes are drawn with heading from nuscenes_infos_val.pkl (the dump
GT table stores no yaw; gt_idx was verified to index both).

Run:  python plot_fig4.py
Out:  figures/final/fig4_qualitative_bev.{pdf,png}
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bev_case import C_C0, C_GT, C_S0, SCORE_THR, draw_case  # noqa: E402
from figstyle import save  # noqa: E402

CASES = [
    dict(token='1fa5506ca31d4174955140d2138db679', gt_idx=37, half=8.0,
         title='traffic_cone, 0 pts, $e$ = $-$2.687\nS0 score = 0.29',
         zeropt=True),
    dict(token='1477ef1a6a3b42daa70ac218e6f6c6f9', gt_idx=2, half=10.0,
         title='pedestrian, 1 pt, $e$ = $-$2.046\nS0 score = 0.11',
         zeropt=False),
    dict(token='3218ed6bfae94f36b82c4285d28e64e5', gt_idx=24, half=12.0,
         title='car, 0 pts, $e$ = $-$1.674\nS0 score = 0.12',
         zeropt=True),
]

fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.1))

c0 = s0 = None
import bev_case  # noqa: E402

for ax, case in zip(axes, CASES):
    dumps, s0dump, _ = bev_case._load()
    g = dumps['gts'][case['token']][case['gt_idx']]
    draw_case(ax, case['token'], g['x'], g['y'], case['half'],
              target_gt_idx=case['gt_idx'])
    ax.set_title(f'target GT: {case["title"]}', fontsize=8)
    if case['zeropt']:
        ax.text(0.02, 0.02,
                '0-pt GT: no object points in the input;\n'
                'detection may leverage scene context\n'
                'and class priors',
                transform=ax.transAxes, fontsize=6.4, color='0.3',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', fc='white',
                          ec='0.7', lw=0.5, alpha=0.85), zorder=6)

legend = [Line2D([0], [0], color=C_GT, lw=2, label='GT box'),
          Line2D([0], [0], color=C_C0, lw=1.6, ls='--',
                 label=f'C0 pred (score ≥ {SCORE_THR})'),
          Line2D([0], [0], color=C_S0, lw=1.6,
                 label=f'S0 pred (score ≥ {SCORE_THR})'),
          Line2D([0], [0], marker='*', color='black', ls='none',
                 markersize=9, label='target low-evidence GT'),
          Line2D([0], [0], marker='.', color='0.55', ls='none',
                 markersize=6, label='LiDAR points')]
fig.legend(handles=legend, loc='lower center', ncol=5, fontsize=8,
           frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.suptitle('Representative qualitative examples: low-evidence GTs '
             'recalled by S0 (epoch-20, seed 3407, thr = 0.10, '
             'StrictMatch@2.0)', fontsize=9.5)
fig.tight_layout(rect=[0, 0.105, 1, 0.95])
save(fig, 'fig4_qualitative_bev')
