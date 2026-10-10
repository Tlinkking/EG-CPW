"""Figure 2: evidence-group recall — absolute levels + delta with CI.

Revision vs old fig3_sparse_recall_analysis (bar chart) and v1
fig_stratified_recall (delta-only forest):
  - panel (a) shows absolute C0/S0 recall per group per seed (the delta
    alone hid the 37% vs 97% absolute gap between Low and High);
  - panel (b) keeps the delta forest; High group is explicitly marked
    "CI crosses 0 — not significant" to avoid visual over-claiming;
  - CI type is stated on the figure: 95% per-seed frame-cluster
    bootstrap, 1000 resamples (NOT a cross-seed robustness interval).

Data hardcoded from (verified by grep against the reports):
  seed 3407: work_dirs/evidence_controlled/analysis/
             C0full_vs_S0full_audit_report.txt (lines 5-10, 43-48)
  seed 3408: work_dirs/evidence_controlled_seed3408/analysis/
             C0full_vs_S0full_audit_report.txt (lines 5-10, 43-48)
Protocol: StrictMatch@2.0 m, score thr 0.10, per-class evidence
25/75 quantile groups on the fixed val GT table (offline stratification).

Run:  python plot_fig2.py
Out:  figures/final/fig2_evidence_group_recall.{pdf,png}
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import C_C0, C_S0, GROUP_COLORS, save  # noqa: E402

GROUPS = ['Low', 'Mid', 'High']
N_GT = [47449, 93191, 46888]

# (seed: {group: (C0, S0, delta, ci_lo, ci_hi)})  StrictMatch@2.0
DATA = {
    3407: {'Low': (0.3691, 0.3756, +0.0065, +0.0040, +0.0093),
           'Mid': (0.7967, 0.7999, +0.0032, +0.0017, +0.0046),
           'High': (0.9718, 0.9729, +0.0011, -0.0002, +0.0024)},
    3408: {'Low': (0.3631, 0.3769, +0.0138, +0.0112, +0.0166),
           'Mid': (0.7938, 0.7993, +0.0055, +0.0040, +0.0070),
           'High': (0.9708, 0.9719, +0.0011, -0.0003, +0.0024)},
}
SEED_MARKER = {3407: 'o', 3408: 's'}

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9),
                         gridspec_kw={'width_ratios': [1.15, 1.0]})

# ---- (a) absolute recall --------------------------------------------------
ax = axes[0]
x = np.arange(3)
dodge = {'C0': -0.18, 'S0': +0.18}
seed_off = {3407: -0.06, 3408: +0.06}
for seed in (3407, 3408):
    for method, color in [('C0', C_C0), ('S0', C_S0)]:
        ys = [DATA[seed][g][0 if method == 'C0' else 1] for g in GROUPS]
        xs = x + dodge[method] + seed_off[seed]
        ax.plot(xs, ys, SEED_MARKER[seed], color=color, markersize=4.5,
                ls='none',
                label=None)
        off = (0, 4.5) if seed == 3407 else (0, -8.5)
        va = 'bottom' if seed == 3407 else 'top'
        for xi, yi, g in zip(xs, ys, GROUPS):
            ax.annotate(f'{yi:.3f}', (xi, yi), textcoords='offset points',
                        xytext=off, ha='center', va=va, fontsize=6.0,
                        color=color)
# connect C0->S0 per (seed, group) to show the delta direction
for seed in (3407, 3408):
    for gi, g in enumerate(GROUPS):
        x0 = gi + dodge['C0'] + seed_off[seed]
        x1 = gi + dodge['S0'] + seed_off[seed]
        ax.plot([x0, x1], [DATA[seed][g][0], DATA[seed][g][1]],
                color='0.6', lw=0.7, ls=':', zorder=1)
ax.set_xticks(x)
ax.set_xticklabels([f'{g}\n(n={n:,})' for g, n in zip(GROUPS, N_GT)])
ax.set_ylabel('StrictMatch@2.0 recall (thr = 0.10)')
ax.set_ylim(0.30, 1.06)
ax.set_title('(a) Absolute recall by evidence group')
ax.grid(axis='y')
handles = [plt.Line2D([], [], marker='o', color=C_C0, ls='none',
                      markersize=4.5, label='C0 (baseline)'),
           plt.Line2D([], [], marker='o', color=C_S0, ls='none',
                      markersize=4.5, label='S0 (peak weight)'),
           plt.Line2D([], [], marker='o', color='0.3', ls='none',
                      markersize=4.5, label='seed 3407'),
           plt.Line2D([], [], marker='s', color='0.3', ls='none',
                      markersize=4.5, label='seed 3408')]
ax.legend(handles=handles, loc='upper left', ncol=2, framealpha=0.9,
          columnspacing=0.8, handletextpad=0.4)

# ---- (b) delta forest ------------------------------------------------------
ax = axes[1]
ypos = {('Low', 3407): 5.3, ('Low', 3408): 4.7,
        ('Mid', 3407): 3.3, ('Mid', 3408): 2.7,
        ('High', 3407): 1.3, ('High', 3408): 0.7}
ax.axvline(0.0, color='0.35', lw=0.8)
for seed in (3407, 3408):
    for g in GROUPS:
        _, _, d, lo, hi = DATA[seed][g]
        y = ypos[(g, seed)]
        color = GROUP_COLORS[g]
        ax.plot([lo, hi], [y, y], color=color, lw=1.6,
                solid_capstyle='round')
        ax.plot(d, y, SEED_MARKER[seed], color=color, markersize=5,
                markeredgecolor='black' if seed == 3408 else 'none',
                markeredgewidth=0.5)
        ns = lo < 0 < hi
        ax.annotate(f'{d:+.4f} [{lo:+.4f},{hi:+.4f}]'
                    + (' (CI incl. 0)' if ns else ''), (hi, y),
                    textcoords='offset points', xytext=(4, 0),
                    va='center', fontsize=6.2,
                    color='0.35' if ns else color)
for gi, g in enumerate(GROUPS):
    ymid = (ypos[(g, 3407)] + ypos[(g, 3408)]) / 2
    ax.text(-0.0042, ymid, f'{g}\n(n={N_GT[gi]:,})', ha='right',
            va='center', fontsize=8)
ax.set_xlim(-0.0045, 0.0280)
ax.set_ylim(0.2, 5.8)
ax.set_yticks([])
ax.set_xlabel('Δ recall (S0 − C0), StrictMatch@2.0, thr = 0.10')
ax.set_title('(b) Delta with 95% CI\n(per-seed frame-cluster bootstrap, '
             '1000 resamples)')
ax.grid(axis='x')
# marker meanings are given in panel (a)'s legend; no legend here to
# avoid colliding with the CI value labels

fig.tight_layout()
save(fig, 'fig2_evidence_group_recall')
