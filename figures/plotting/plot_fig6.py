"""Figure 6: evidence-weighting mechanism (4 panels).

Revision vs v1 fig_evidence_weight (2 panels) per reviewer request:
  (a) deterministic evidence->weight mapping with omega=1, omega_max=1.5
      and the e=-q_max=-2.0 saturation point annotated;
  (b) evidence-score distribution by group (unchanged from v1);
  (c) count-density scatter colored by weight (unchanged from v1);
  (d) per-class weight statistics (median weight and share of GTs with
      w>1) — new, shows which classes carry the up-weighting.

Caption must state: this figure explains the weight ASSIGNMENT
mechanism only; it does not by itself establish a causal performance
effect.

Data: ../../data/evidence_scatter.npz (val GT table, seed-3407 dump;
identical for C0/S0). Mapping parameters from the config
(lambda=0.25, q_max=2.0, omega_max=1.5).

Run:  python plot_fig6.py
Out:  figures/final/fig6_evidence_weight_mechanism.{pdf,png}
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import (C_C0, C_S0, CLASS_NAMES, GROUP_COLORS,  # noqa: E402
                      save)

DATA = Path(__file__).resolve().parents[2] / 'data' / 'evidence_scatter.npz'
GROUPS = ['Low', 'Mid', 'High']
LAMBDA, Q_MAX, W_MAX = 0.25, 2.0, 1.5

d = np.load(DATA)
num_pts = d['num_pts'].astype(np.float64)
volume = d['volume'].astype(np.float64)
cls_id = d['cls_id'].astype(int)
evid = d['evidence'].astype(np.float64)
group = d['group']
weight = np.clip(1.0 + LAMBDA * np.clip(-evid, 0.0, Q_MAX), 1.0, W_MAX)

fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))

# ---- (a) mapping curve -----------------------------------------------------
ax = axes[0, 0]
e = np.linspace(-3.0, 3.0, 1201)
w = np.clip(1.0 + LAMBDA * np.clip(-e, 0.0, Q_MAX), 1.0, W_MAX)
ax.plot(e, w, color=C_S0, lw=1.8)
ax.axhline(1.0, color='gray', lw=0.8, ls=':')
ax.axhline(W_MAX, color=C_C0, lw=0.8, ls='--')
ax.axvline(-Q_MAX, color=C_C0, lw=0.8, ls='--')
ax.annotate('$\\omega_{max}$ = 1.5', xy=(-2.6, 1.5), xytext=(-0.9, 1.43),
            fontsize=7.5, color=C_C0,
            arrowprops=dict(arrowstyle='->', color=C_C0, lw=0.8))
ax.annotate('saturation\n$e$ = $-q_{max}$ = $-$2.0', xy=(-2.0, 1.5),
            xytext=(-2.95, 1.24), fontsize=7, ha='left', color=C_C0,
            arrowprops=dict(arrowstyle='->', color=C_C0, lw=0.8))
ax.annotate('$\\omega$ = 1 (no down-weighting)', xy=(2.4, 1.0),
            xytext=(0.55, 1.10), fontsize=7, color='0.3',
            arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
ax.text(-0.5, 1.21, '$\\omega_i$ = clip(1 + $\\lambda$\u2009clip($-e_i$,'
        ' 0, $q_{max}$), 1, $\\omega_{max}$)\n$\\lambda$ = 0.25, '
        '$q_{max}$ = 2.0, $\\omega_{max}$ = 1.5',
        fontsize=7.5, ha='center',
        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='0.6',
                  lw=0.6))
ax.set_xlabel('evidence score $e_i$')
ax.set_ylabel('center-peak weight $\\omega_i$')
ax.set_xlim(-3, 3)
ax.set_ylim(0.96, 1.60)
ax.set_title('(a) Evidence-to-weight mapping (training-time, S0)')
ax.grid()

# ---- (b) evidence distribution --------------------------------------------
ax = axes[0, 1]
bins = np.linspace(-3.0, 3.0, 121)
for gi, gname in enumerate(GROUPS):
    e_g = evid[group == gi]
    ax.hist(e_g, bins=bins, density=True, histtype='stepfilled',
            alpha=0.45, color=GROUP_COLORS[gname],
            label=f'{gname} (n={e_g.size:,})')
    ax.hist(e_g, bins=bins, density=True, histtype='step',
            color=GROUP_COLORS[gname], lw=1.0)
ax.axvline(-Q_MAX, color=C_C0, ls='--', lw=0.9)
ax.axvline(0.0, color='gray', ls=':', lw=0.9)
ax.text(-Q_MAX - 0.06, ax.get_ylim()[1] * 0.97, '$e=-q_{max}$',
        ha='right', va='top', fontsize=7, color=C_C0)
ax.set_xlabel('evidence score $e_i$ (val GT)')
ax.set_ylabel('density')
ax.set_title('(b) Evidence score distribution by group\n'
             '(per-class 25th/75th percentile partition; boundaries '
             'differ by class)')
ax.legend(fontsize=7, loc='upper right')
ax.set_xlim(-3, 3)

# ---- (c) count-density scatter --------------------------------------------
ax = axes[1, 0]
rng = np.random.RandomState(3407)
n_sub = 20000
idx = rng.choice(num_pts.size, size=n_sub, replace=False)
density = num_pts / np.maximum(volume, 1e-6)
sc = ax.scatter(np.log10(1.0 + num_pts[idx]),
                np.log10(1.0 + density[idx]),
                c=weight[idx], cmap='viridis', vmin=1.0, vmax=W_MAX,
                s=2.5, alpha=0.35, rasterized=True)
cb = fig.colorbar(sc, ax=ax, pad=0.02)
cb.set_label('weight $\\omega_i$', fontsize=7.5)
cb.ax.tick_params(labelsize=7)
ax.set_xlabel('log$_{10}$(1 + enclosed point count)')
ax.set_ylabel('log$_{10}$(1 + points / m$^3$)')
ax.set_title('(c) Count–density scatter colored by weight\n'
             f'(subsample n={n_sub:,} of {num_pts.size:,} GTs)')
ax.grid()

# ---- (d) per-class weight statistics ---------------------------------------
ax = axes[1, 1]
med_w, share_up, n_cls = [], [], []
for cid in range(10):
    sel = cls_id == cid
    med_w.append(float(np.median(weight[sel])))
    share_up.append(float((weight[sel] > 1.0).mean()))
    n_cls.append(int(sel.sum()))
order = np.argsort(med_w)
y = np.arange(10)
labels = [f'{CLASS_NAMES[i]} (n={n_cls[i]:,})' for i in order]
ax.plot(np.array(med_w)[order], y, 'o', color=C_S0, markersize=5,
        label='median weight $\\omega_i$')
ax2 = ax.twiny()
ax2.plot(np.array(share_up)[order], y, 's', color=C_C0, markersize=4.5,
         label='share of GTs with $\\omega_i$ > 1')
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=7)
ax.set_xlabel('median weight', color=C_S0)
ax2.set_xlabel('share with $\\omega_i$ > 1', color=C_C0)
ax.tick_params(axis='x', labelcolor=C_S0)
ax2.tick_params(axis='x', labelcolor=C_C0)
ax.set_xlim(0.98, 1.52)
ax2.set_xlim(-0.02, 1.02)
ax.set_title('(d) Per-class weight statistics (val GT table)')
ax.grid(axis='y')
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=6.5, loc='center left',
          bbox_to_anchor=(0.0, 0.32), framealpha=0.95)

fig.tight_layout()
save(fig, 'fig6_evidence_weight_mechanism')
