"""Figure S1 (supplementary; revised Figure 5):
Score-threshold sweep — group-restricted recall vs overall precision.

Moved to supplementary per revision plan. Accurate definitions:
  - recall: group-restricted StrictMatch@2.0 recall (group GT
    denominator);
  - precision: overall prediction-level precision at the same
    threshold (NOT a per-group precision);
  - panel (c): paired per-threshold recall delta with 95% frame-cluster
    bootstrap CI (1000 resamples over frames, shared resample indices).

Data:
  (a)(b) work_dirs/evidence_controlled/threshold_control_analysis/
         tables/evidence_group_curve.md (parsed, not retyped);
  (c)    ../../data/sweep_matched_seed{3407,3408}.npz (round-2 export;
         StrictMatch@2.0 recomputed per threshold on the fixed dumps).

Run:  python plot_fig5.py
Out:  supplementary_figures/figS1_threshold_sweep.{pdf,png}
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import C_C0, C_S0, GROUP_COLORS, save  # noqa: E402

CURVE_MD = (Path(__file__).resolve().parents[3] /
            'threshold_control_analysis/tables/evidence_group_curve.md')
DATA_DIR = Path(__file__).resolve().parents[2] / 'data'
N_BOOT = 1000
BOOT_SEED = 3407
GROUPS = ['Low', 'Mid', 'High']


def parse_curve_md(path):
    """Parse 'thr | method | overall recall | overall precision |
    Low | Mid | High' rows per seed block."""
    out = {}
    seed = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith('## seed'):
            seed = int(line.replace('## seed', ''))
            out[seed] = []
        elif line.startswith('| 0.') and seed is not None:
            parts = [p.strip() for p in line.strip('|').split('|')]
            if len(parts) >= 7 and parts[1] in ('C0', 'S0'):
                out[seed].append(dict(
                    thr=float(parts[0]), method=parts[1],
                    recall=float(parts[2]), precision=float(parts[3]),
                    low=float(parts[4]), mid=float(parts[5]),
                    high=float(parts[6])))
    return out


curve = parse_curve_md(CURVE_MD)
fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.3))
marker = {3407: 'o', 3408: 's'}

# ---- (a) overall PR operating curve ---------------------------------------
ax = axes[0]
for seed in (3407, 3408):
    for method, color in [('C0', C_C0), ('S0', C_S0)]:
        rows = [r for r in curve[seed] if r['method'] == method]
        ax.plot([r['recall'] for r in rows],
                [r['precision'] for r in rows],
                '-' if seed == 3407 else '--', color=color, lw=1.3,
                marker=marker[seed], markersize=3)
        for r in rows:
            if r['thr'] in (0.10, 0.40):
                ax.annotate(f"{r['thr']:.2f}", (r['recall'],
                            r['precision']), textcoords='offset points',
                            xytext=(3, 3), fontsize=6, color=color)
ax.set_xlabel('overall recall (StrictMatch@2.0)')
ax.set_ylabel('overall precision')
ax.set_title('(a) Overall PR operating curve\n(thr sweep 0.10–0.80)')
ax.grid()
ax.legend(handles=[
    plt.Line2D([], [], color=C_C0, ls='-', label='C0, seed 3407'),
    plt.Line2D([], [], color=C_S0, ls='-', label='S0, seed 3407'),
    plt.Line2D([], [], color=C_C0, ls='--', label='C0, seed 3408'),
    plt.Line2D([], [], color=C_S0, ls='--', label='S0, seed 3408')],
    fontsize=6.4, loc='upper right')

# ---- (b) group-restricted recall vs overall precision ----------------------
ax = axes[1]
for seed in (3407, 3408):
    for gname in GROUPS:
        key = gname.lower()
        for method, color, mfill in [('C0', C_C0, 'none'),
                                     ('S0', C_S0, None)]:
            rows = [r for r in curve[seed] if r['method'] == method]
            ax.plot([r[key] for r in rows], [r['precision'] for r in rows],
                    '-' if seed == 3407 else '--',
                    color=GROUP_COLORS[gname],
                    lw=1.3 if method == 'S0' else 1.1,
                    marker=marker[seed], markersize=2.8,
                    markerfacecolor=mfill or GROUP_COLORS[gname],
                    alpha=1.0 if method == 'S0' else 0.75)
ax.set_xlabel('group-restricted recall (StrictMatch@2.0)')
ax.set_ylabel('overall precision (same threshold)')
ax.set_title('(b) Group-restricted recall vs overall precision\n'
             '(precision is NOT per-group; filled = S0, open = C0)')
ax.grid()
ax.legend(handles=[
    plt.Line2D([], [], color=GROUP_COLORS[g], ls='-', label=g)
    for g in GROUPS] + [
    plt.Line2D([], [], color='0.3', ls='-', label='seed 3407'),
    plt.Line2D([], [], color='0.3', ls='--', label='seed 3408')],
    fontsize=6.4, loc='lower left')

# ---- (c) per-threshold recall delta with bootstrap CI ----------------------
ax = axes[2]
ax.axhline(0.0, color='0.4', lw=0.8)
for seed in (3407, 3408):
    d = np.load(DATA_DIR / f'sweep_matched_seed{seed}.npz')
    thr = d['thresholds']
    tokens, frame_of = np.unique(d['token'], return_inverse=True)
    F = len(tokens)
    T = len(thr)
    boot_idx = np.random.RandomState(BOOT_SEED).randint(
        0, F, size=(N_BOOT, F))
    for gi, gname in enumerate(['Overall', 'Low', 'Mid', 'High']):
        sel = np.ones(len(d['group']), dtype=bool) if gi == 0 \
            else d['group'] == (gi - 1)
        color = '0.25' if gi == 0 else GROUP_COLORS[gname]
        idx = np.where(sel)[0]
        tot = np.zeros((F, T), dtype=np.int64)
        dm = np.zeros((F, T), dtype=np.int64)
        np.add.at(tot, (frame_of[idx],), 1)
        # per-threshold matched-delta counts per frame
        for ti in range(T):
            np.add.at(dm[:, ti], frame_of[idx],
                      d['matched_s0'][idx, ti].astype(np.int64)
                      - d['matched_c0'][idx, ti].astype(np.int64))
        tot_f = tot[:, 0]  # group size is threshold-independent
        point = dm.sum(0) / tot_f.sum()
        t_boot = tot_f[boot_idx].sum(1)
        lo = np.empty(T)
        hi = np.empty(T)
        for ti in range(T):
            s = dm[:, ti][boot_idx].sum(1) / t_boot
            lo[ti], hi[ti] = np.percentile(s, [2.5, 97.5])
        ls = '-' if seed == 3407 else '--'
        ax.plot(thr, point, ls, color=color, lw=1.3,
                marker=marker[seed], markersize=2.8)
        ax.fill_between(thr, lo, hi, color=color,
                        alpha=0.16 if seed == 3407 else 0.10, lw=0)
ax.set_xlabel('score threshold')
ax.set_ylabel('Δ recall (S0 − C0), StrictMatch@2.0')
ax.set_title('(c) Recall delta vs threshold\n'
             '(95% frame-cluster bootstrap CI, 1000 resamples)')
ax.grid()
ax.legend(handles=[
    plt.Line2D([], [], color='0.25', ls='-', label='Overall')] + [
    plt.Line2D([], [], color=GROUP_COLORS[g], ls='-', label=g)
    for g in GROUPS] + [
    plt.Line2D([], [], color='0.3', ls='-', label='seed 3407'),
    plt.Line2D([], [], color='0.3', ls='--', label='seed 3408')],
    fontsize=6.4, loc='upper right')

fig.tight_layout()
save(fig, 'figS1_threshold_sweep', supplementary=True)
