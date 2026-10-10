"""Shared BEV case-drawing helper for fig4 / figS2.

Loads the epoch-20 seed-3407 dumps and nuScenes val infos once and
draws a windowed BEV panel: raw LiDAR points (gray, rasterized), GT
boxes, C0/S0 predictions (score >= 0.10), plus a highlighted target
(GT or FP box). Colorblind-safe Okabe-Ito palette with linestyle
redundancy: GT = bluish-green solid, C0 = vermillion dashed,
S0 = blue solid.
"""

import pickle
from pathlib import Path

import numpy as np
from matplotlib.patches import Circle, Polygon

ROOT = Path('/home/bjtc/projects_ws/mmdetection3d-1.3.0')
DUMP_C0 = ROOT / 'work_dirs/evidence_controlled/analysis/C0full_dump.pkl'
DUMP_S0 = ROOT / 'work_dirs/evidence_controlled/analysis/S0full_dump.pkl'
INFOS = Path('/home/bjtc/data/datasets/nuScenes/nuscenes_infos_val.pkl')
LIDAR_DIR = Path('/home/bjtc/data/datasets/nuScenes/samples/LIDAR_TOP')

C_GT = '#009E73'
C_C0 = '#D55E00'
C_S0 = '#0072B2'
C_HL = '#CC79A7'

SCORE_THR = 0.10
MAX_PTS = 80000

_cache = {}


def _load():
    if 'c0' not in _cache:
        with open(DUMP_C0, 'rb') as f:
            _cache['c0'] = pickle.load(f)
        with open(DUMP_S0, 'rb') as f:
            _cache['s0'] = pickle.load(f)
        with open(INFOS, 'rb') as f:
            infos = pickle.load(f)['data_list']
        _cache['tok2info'] = {i['token']: i for i in infos}
    return _cache['c0'], _cache['s0'], _cache['tok2info']


def box_corners(x, y, dx, dy, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.array([[dx / 2, dy / 2], [dx / 2, -dy / 2],
                      [-dx / 2, -dy / 2], [-dx / 2, dy / 2]])
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([x, y])


def draw_box(ax, x, y, dx, dy, yaw, color, lw=1.2, ls='-', alpha=1.0,
             zorder=3):
    pts = box_corners(x, y, dx, dy, yaw)
    ax.add_patch(Polygon(pts, closed=True, fill=False, edgecolor=color,
                         lw=lw, ls=ls, alpha=alpha, zorder=zorder))


def draw_case(ax, token, cx, cy, half, target_gt_idx=None,
              fp_xy=None, fp_score=None, seed=3407):
    """Draw one BEV window centered at (cx, cy).

    target_gt_idx: index of the target GT in the dump GT table
        (highlighted with star + dashed circle).
    fp_xy / fp_score: highlight an S0-only FP box instead
        (magenta ring + score annotation).
    """
    c0, s0, tok2info = _load()
    info = tok2info[token]
    insts = info['instances']
    rng = np.random.RandomState(seed)

    pts = np.fromfile(LIDAR_DIR / info['lidar_points']['lidar_path'],
                      dtype=np.float32).reshape(-1, 5)
    m = (np.abs(pts[:, 0] - cx) < half) & (np.abs(pts[:, 1] - cy) < half)
    pts = pts[m]
    if pts.shape[0] > MAX_PTS:
        pts = pts[rng.choice(pts.shape[0], MAX_PTS, replace=False)]
    ax.scatter(pts[:, 0], pts[:, 1], s=1.2, c='0.55', rasterized=True,
               zorder=1)

    for j, gt in enumerate(c0['gts'][token]):
        if abs(gt['x'] - cx) > half or abs(gt['y'] - cy) > half:
            continue
        yaw = insts[j]['bbox_3d'][6]
        tgt = (j == target_gt_idx)
        draw_box(ax, gt['x'], gt['y'], gt['sx'], gt['sy'], yaw, C_GT,
                 lw=2.2 if tgt else 1.2, alpha=1.0 if tgt else 0.55)

    for dump, color, ls in [(c0, C_C0, '--'), (s0, C_S0, '-')]:
        p = dump['preds'][token]
        boxes, scores = p['boxes'], p['scores']
        m = (scores >= SCORE_THR) & (np.abs(boxes[:, 0] - cx) < half) & \
            (np.abs(boxes[:, 1] - cy) < half)
        for b, sc in zip(boxes[m], scores[m]):
            draw_box(ax, b[0], b[1], b[3], b[4], b[6], color, lw=1.2,
                     ls=ls, alpha=0.9)

    if target_gt_idx is not None:
        ax.add_patch(Circle((cx, cy), radius=half * 0.16, fill=False,
                            edgecolor='black', lw=1.3, ls='--', zorder=4))
        ax.plot(cx, cy, marker='*', color='black', markersize=9,
                zorder=5)
    if fp_xy is not None:
        ax.add_patch(Circle(fp_xy, radius=half * 0.14, fill=False,
                            edgecolor=C_HL, lw=1.6, ls='-', zorder=4))
        ax.annotate(f'S0-only FP\nscore={fp_score:.2f}', xy=fp_xy,
                    xytext=(cx - half * 0.97, cy + half * 0.86),
                    fontsize=7, color=C_HL,
                    arrowprops=dict(arrowstyle='->', color=C_HL, lw=0.9),
                    zorder=5)

    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect('equal')
    ax.set_xlabel('x [m, LiDAR frame]', fontsize=7.5)
    ax.set_ylabel('y [m, LiDAR frame]', fontsize=7.5)
    ax.tick_params(labelsize=6.5)
