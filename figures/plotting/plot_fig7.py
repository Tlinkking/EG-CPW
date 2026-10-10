"""Figure 7: gained TP vs gained FP comparison (offline diagnostic).

Revision vs v1 fig_failure_analysis: adds the gained-TP side so the
trade-off is shown symmetrically.

Definitions (thr = 0.10, StrictMatch@2.0, epoch-20 dumps):
  gained TP: GT matched by S0 but not by C0 (S0-only true positive;
             the panel uses the matching S0 prediction's score and
             distance);
  gained FP: S0-only false positive = S0 prediction, score >= 0.10,
             unmatched, with no same-class C0 prediction within 2.0 m;
  background: no same-class GT in the frame at all.

Panels:
  (a) score distributions of gained TP vs gained FP (seed 3407);
  (b) distance to nearest same-class GT center: matched distance for
      gained TP (<= 2 m by construction) and CDF for gained FP
      (finite distances; background share annotated);
  (c) class composition (share within each set, seed 3407);
  (d) evidence-group composition of gained TP (incl. zero-point share)
      and context split of gained FP, seeds 3407/3408.

Data: ../../data/gained_tp_seed{3407,3408}.npz and
fp_distance_seed{3407,3408}.npz (round-2 export; counts asserted
identical to the earlier audits).

Run:  python plot_fig7.py
Out:  figures/final/fig7_gained_tp_vs_fp.{pdf,png}
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import (CLASS_NAMES, C_C0, C_ORANGE, C_PINK, C_S0,  # noqa: E402
                      GROUP_COLORS, save)

DATA_DIR = Path(__file__).resolve().parents[2] / 'data'
GROUPS = ['Low', 'Mid', 'High']

g7 = np.load(DATA_DIR / 'gained_tp_seed3407.npz')
g8 = np.load(DATA_DIR / 'gained_tp_seed3408.npz')
f7 = np.load(DATA_DIR / 'fp_distance_seed3407.npz')
f8 = np.load(DATA_DIR / 'fp_distance_seed3408.npz')

fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.8))

# ---- (a) score distributions ----------------------------------------------
ax = axes[0, 0]
bins = np.linspace(0.10, 1.0, 46)
ax.hist(f7['s0only_score'], bins=bins, density=True, histtype='step',
        lw=1.3, color=C_PINK,
        label=f"gained FP (n={f7['s0only_score'].size:,})")
ax.hist(g7['s0_score'], bins=bins, density=True, histtype='step',
        lw=1.6, color=C_S0,
        label=f"gained TP (n={g7['s0_score'].size:,})")
ax.set_yscale('log')
ax.set_xlabel('S0 prediction score')
ax.set_ylabel('density (log)')
ax.set_title('(a) Score distribution, seed 3407')
ax.legend(fontsize=7, loc='upper right')
ax.grid()

# ---- (b) distance to nearest same-class GT ---------------------------------
ax = axes[0, 1]
for fp, seed, alpha in [(f7, 3407, 1.0), (f8, 3408, 0.55)]:
    dist = fp['s0only_dist']
    fin = np.sort(dist[np.isfinite(dist)])
    ys = np.arange(1, fin.size + 1) / fin.size
    ax.plot(fin, ys, color=C_PINK, lw=1.4, alpha=alpha)
for g, seed, alpha in [(g7, 3407, 1.0), (g8, 3408, 0.55)]:
    xs = np.sort(g['s0_dist'])
    ys = np.arange(1, xs.size + 1) / xs.size
    ax.plot(xs, ys, color=C_S0, lw=1.4, alpha=alpha)
ax.axvline(2.0, color='0.4', ls=':', lw=0.9)
ax.text(2.35, 0.30, 'match cap 2 m', fontsize=6.5, color='0.4',
        rotation=90, va='bottom')
ax.text(0.97, 0.04,
        'faint = seed 3408\n'
        f'background share (gained FP, no same-class GT in frame):\n'
        f'3407: {float(np.isinf(f7["s0only_dist"]).mean()):.1%}  '
        f'3408: {float(np.isinf(f8["s0only_dist"]).mean()):.1%}',
        transform=ax.transAxes, fontsize=6.4, va='bottom', ha='right',
        color='0.3')
ax.set_xscale('log')
ax.set_xticks([0.1, 0.5, 1, 2, 5, 10, 20, 50])
ax.set_xticklabels(['0.1', '0.5', '1', '2', '5', '10', '20', '50'])
ax.set_xlabel('distance to nearest same-class GT center [m, log]')
ax.set_ylabel('CDF')
ax.set_title('(b) Distance to nearest same-class GT\n'
             '(CDF over finite distances)')
ax.legend(handles=[
    plt.Line2D([], [], color=C_PINK, lw=1.4,
               label=f'gained FP (n={f7["s0only_dist"].size:,}, seed 3407)'),
    plt.Line2D([], [], color=C_S0, lw=1.4,
               label=f'gained TP match dist (n={g7["s0_score"].size:,}, '
                     'seed 3407)')],
    fontsize=6.4, loc='upper left')
ax.set_ylim(0, 1.02)
ax.grid()

# ---- (c) class composition --------------------------------------------------
ax = axes[1, 0]
tp_cnt = np.bincount(g7['cls_id'], minlength=10)
fp_cnt = np.bincount(f7['s0only_cls'], minlength=10)
order = np.argsort(tp_cnt / tp_cnt.sum())
y = np.arange(10)
for cnt, name, c, mk in [(tp_cnt, 'gained TP', C_S0, 'o'),
                         (fp_cnt, 'gained FP', C_PINK, 's')]:
    share = cnt / cnt.sum()
    ax.hlines(y, 0, share[order], color=c, alpha=0.22, lw=1.1)
    ax.plot(share[order], y, mk, color=c, markersize=4.5, ls='none',
            label=name)
ax.set_yticks(y)
ax.set_yticklabels([CLASS_NAMES[i] for i in order], fontsize=7)
ax.set_xlabel('share within set')
ax.set_title('(c) Class composition, seed 3407')
ax.legend(fontsize=7, loc='lower right')
ax.set_xlim(left=0)
ax.grid(axis='y')

# ---- (d) group / context composition ---------------------------------------
ax = axes[1, 1]
rows = []
for g, seed in [(g7, 3407), (g8, 3408)]:
    grp_share = [(g['group'] == gi).mean() for gi in range(3)]
    zp = float((g['num_pts'] == 0).mean())
    rows.append((f'gained TP, seed {seed}\n(n={len(g["group"]):,})',
                 grp_share, zp))
for fp, seed in [(f7, 3407), (f8, 3408)]:
    bg = float(np.isinf(fp['s0only_dist']).mean())
    rows.append((f'gained FP, seed {seed}\n(n={fp["s0only_dist"].size:,})',
                 None, bg))

y = np.arange(4)[::-1]
for (label, grp_share, extra), yi in zip(rows, y):
    if grp_share is not None:
        for gi, gname in enumerate(GROUPS):
            ax.plot(grp_share[gi], yi + 0.22 - gi * 0.22, 'o',
                    color=GROUP_COLORS[gname], markersize=5)
            ax.annotate(f'{gname} {grp_share[gi]:.1%}',
                        (grp_share[gi], yi + 0.22 - gi * 0.22),
                        textcoords='offset points', xytext=(6, -1.5),
                        fontsize=6.4, color=GROUP_COLORS[gname])
        ax.annotate(f'zero-point GT share: {extra:.1%}',
                    (0.985, yi - 0.33), xycoords=('axes fraction',
                    'data'), ha='right', fontsize=6.4, color='0.35')
    else:
        ax.plot(1 - extra, yi + 0.11, 'o', color=C_ORANGE, markersize=5)
        ax.annotate(f'same-class GT in frame {1 - extra:.1%}',
                    (1 - extra, yi + 0.11), textcoords='offset points',
                    xytext=(6, -1.5), fontsize=6.4, color=C_ORANGE)
        ax.plot(extra, yi - 0.11, 's', color='0.45', markersize=5)
        ax.annotate(f'background (no same-class GT) {extra:.1%}',
                    (extra, yi - 0.11), textcoords='offset points',
                    xytext=(6, -1.5), fontsize=6.4, color='0.45')
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows], fontsize=7)
ax.set_xlim(0, 1.05)
ax.set_ylim(-0.6, 3.6)
ax.set_xlabel('share within set')
ax.set_title('(d) Evidence-group / context composition\n'
             '(both seeds)')
ax.grid(axis='x')

fig.tight_layout()
save(fig, 'fig7_gained_tp_vs_fp')
