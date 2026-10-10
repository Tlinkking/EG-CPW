"""Figure 3: recall vs GT center distance — delta with paired bootstrap CI.

Revision vs v1 fig_distance_recall (absolute curves only):
  - bottom row: S0-C0 recall delta per 5 m distance bin with a paired
    95% frame-cluster bootstrap CI (1000 resamples over frames, same
    resample indices shared across bins; filled marker = CI excludes 0);
  - top row: absolute C0/S0 recall curves kept as auxiliary panels;
  - every bin is annotated with its GT count; bins with n < 30 omitted.

Data: ../../data/distance_recall_seed{3407,3408}.npz (round-2 export,
includes per-GT frame tokens). StrictMatch@2.0, score thr 0.10,
per-class evidence 25/75 quantile groups on the fixed val GT table.

Run:  python plot_fig3.py
Out:  figures/final/fig3_distance_recall_delta.{pdf,png}
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import C_C0, C_S0, GROUP_COLORS, save  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[2] / 'data'
BIN_LO, BIN_HI, BIN_W = 0.0, 75.0, 5.0
MIN_N = 30
N_BOOT = 1000
BOOT_SEED = 3407
PANELS = ['Overall', 'Low', 'Mid', 'High']

edges = np.arange(BIN_LO, BIN_HI + 1e-6, BIN_W)
centers = edges[:-1] + BIN_W / 2
B = len(centers)


def frame_matrices(d, sel):
    """Per-frame per-bin totals and matched counts for selection sel."""
    tokens, frame_of = np.unique(d['token'], return_inverse=True)
    F = len(tokens)
    bin_of = np.clip(((d['lidar_range'] - BIN_LO) / BIN_W).astype(int),
                     0, B - 1)
    tot = np.zeros((F, B), dtype=np.int64)
    m0 = np.zeros((F, B), dtype=np.int64)
    m1 = np.zeros((F, B), dtype=np.int64)
    idx = np.where(sel)[0]
    np.add.at(tot, (frame_of[idx], bin_of[idx]), 1)
    hit0 = idx[d['matched_c0'][idx]]
    hit1 = idx[d['matched_s0'][idx]]
    np.add.at(m0, (frame_of[hit0], bin_of[hit0]), 1)
    np.add.at(m1, (frame_of[hit1], bin_of[hit1]), 1)
    return tot, m0, m1


def boot_ci(tot, m0, m1, boot_idx):
    """Paired frame-cluster bootstrap CI of the recall delta per bin."""
    F = tot.shape[0]
    delta = np.full((N_BOOT, B), np.nan)
    for b in range(B):
        t = tot[:, b][boot_idx].sum(axis=1)
        d = (m1[:, b][boot_idx].sum(axis=1)
             - m0[:, b][boot_idx].sum(axis=1))
        ok = t > 0
        delta[ok, b] = d[ok] / t[ok]
    lo = np.nanpercentile(delta, 2.5, axis=0)
    hi = np.nanpercentile(delta, 97.5, axis=0)
    point = np.where(tot.sum(0) > 0,
                     (m1.sum(0) - m0.sum(0)) / np.maximum(tot.sum(0), 1),
                     np.nan)
    return point, lo, hi


results = {}  # (seed, panel) -> dict
for seed in (3407, 3408):
    d = np.load(DATA_DIR / f'distance_recall_seed{seed}.npz',
                allow_pickle=False)
    n_frames = len(np.unique(d['token']))
    boot_idx = np.random.RandomState(BOOT_SEED).randint(
        0, n_frames, size=(N_BOOT, n_frames))
    for pi, pname in enumerate(PANELS):
        sel = np.ones(len(d['group']), dtype=bool) if pi == 0 \
            else d['group'] == (pi - 1)
        tot, m0, m1 = frame_matrices(d, sel)
        point, lo, hi = boot_ci(tot, m0, m1, boot_idx)
        with np.errstate(invalid='ignore', divide='ignore'):
            r0 = m0.sum(0) / tot.sum(0)
            r1 = m1.sum(0) / tot.sum(0)
        results[(seed, pname)] = dict(tot=tot.sum(0), r0=r0, r1=r1,
                                      delta=point, lo=lo, hi=hi)

fig, axes = plt.subplots(2, 4, figsize=(11.4, 4.9), sharex=True)
marker = {3407: 'o', 3408: 's'}
ls = {3407: '-', 3408: '--'}

for pi, pname in enumerate(PANELS):
    ax = axes[0, pi]
    for seed in (3407, 3408):
        r = results[(seed, pname)]
        ok = r['tot'] >= MIN_N
        ax.plot(centers[ok], r['r0'][ok], ls[seed], color=C_C0,
                lw=1.1 if seed == 3407 else 0.9,
                alpha=1.0 if seed == 3407 else 0.65)
        ax.plot(centers[ok], r['r1'][ok], ls[seed], color=C_S0,
                lw=1.1 if seed == 3407 else 0.9,
                alpha=1.0 if seed == 3407 else 0.65)
    ax.set_title(f'{pname}', fontsize=8.5)
    ax.set_ylim(0, 1.02)
    ax.grid()
    if pi == 0:
        ax.set_ylabel('recall (StrictMatch@2, thr=0.10)')
        ax.legend(handles=[
            plt.Line2D([], [], color=C_C0, ls='-', label='C0, seed 3407'),
            plt.Line2D([], [], color=C_S0, ls='-', label='S0, seed 3407'),
            plt.Line2D([], [], color=C_C0, ls='--', lw=0.9,
                       label='C0, seed 3408'),
            plt.Line2D([], [], color=C_S0, ls='--', lw=0.9,
                       label='S0, seed 3408')], fontsize=6.4,
            loc='lower left')

    ax = axes[1, pi]
    ax.axhline(0.0, color='0.4', lw=0.8)
    for seed in (3407, 3408):
        r = results[(seed, pname)]
        ok = r['tot'] >= MIN_N
        off = -0.7 if seed == 3407 else 0.7
        color = GROUP_COLORS[pname] if pi > 0 else '0.25'
        for b in np.where(ok)[0]:
            filled = r['lo'][b] > 0 or r['hi'][b] < 0
            ax.plot([centers[b] + off] * 2, [r['lo'][b], r['hi'][b]],
                    color=color, lw=1.0,
                    alpha=0.95 if seed == 3407 else 0.6)
            ax.plot(centers[b] + off, r['delta'][b], marker[seed],
                    color=color, markersize=3.6,
                    markerfacecolor=color if filled else 'white',
                    markeredgewidth=0.9,
                    alpha=0.95 if seed == 3407 else 0.7)
    ax.set_ylabel('Δ recall (S0 − C0)')
    ax.set_xlabel('BEV center distance [m]')
    ax.grid()
    ax.set_ylim(-0.10, 0.16)
    if pi == 0:
        ax.legend(handles=[
            plt.Line2D([], [], marker='o', color='0.25', ls='none',
                       markersize=4, label='seed 3407'),
            plt.Line2D([], [], marker='s', color='0.25', ls='none',
                       markersize=4, alpha=0.7, label='seed 3408'),
            plt.Line2D([], [], marker='o', color='0.25', ls='none',
                       markersize=4, markerfacecolor='white',
                       label='95% CI includes 0'),
            plt.Line2D([], [], marker='o', color='0.25', ls='none',
                       markersize=4, label='95% CI excludes 0')],
            fontsize=6.2, loc='upper right', ncol=2, columnspacing=0.7,
            handletextpad=0.3)

axes[1, 0].texts.clear()
ax_over = axes[1, 0]
r = results[(3407, 'Overall')]
ok = r['tot'] >= MIN_N
ymin, _ = ax_over.get_ylim()
for k, b in enumerate(np.where(ok)[0]):
    ax_over.annotate(f"{int(r['tot'][b]) // 1000}k"
                     if r['tot'][b] >= 1000 else f"{int(r['tot'][b])}",
                     (centers[b], ymin), textcoords='offset points',
                     xytext=(0, 1 if k % 2 == 0 else 6.5), ha='center',
                     fontsize=5.6, color='0.45')

fig.suptitle('Recall vs GT center distance: absolute recall (top) and '
             'S0−C0 delta with paired 95% frame-cluster bootstrap CI '
             '(bottom; bin labels = GT count, n < 30 omitted)',
             fontsize=9)
fig.tight_layout(rect=[0, 0, 1, 0.95])
save(fig, 'fig3_distance_recall_delta')
