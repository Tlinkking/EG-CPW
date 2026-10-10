"""Export real dump-derived data for the paper figure refresh.

Reads the existing epoch-20 dumps only; no inference, no metric
recomputation beyond the established offline protocol (score>=0.1,
StrictMatch@2 greedy one-to-one class-aware matching, per-class
evidence quantile grouping on validation GT — identical to
tools/analysis/evidence_stratified_analysis.py and
proposal_nms_analysis_epoch20.py, whose helper functions are imported
verbatim).

Outputs (NPZ, per seed where applicable) into ../data/:

  distance_recall_seed{3407,3408}.npz
      lidar_range  (N,) float32  GT BEV distance to ego (from dump gts)
      group        (N,) int8     0=Low 1=Mid 2=High (val-derived, per-class
                                 evidence 25/75 quantiles, identical to
                                 the paper stratification)
      cls_id       (N,) int8
      matched_c0   (N,) bool     StrictMatch@2 at thr=0.1 (C0)
      matched_s0   (N,) bool     StrictMatch@2 at thr=0.1 (S0)

  evidence_scatter.npz  (seed-independent GT table, from 3407 C0 dump;
      num_pts      (N,) int32    num_lidar_pts (9+1 sweep aggregated)
      volume       (N,) float32
      cls_id       (N,) int8
      evidence     (N,) float32  e = beta*z_count + (1-beta)*z_density
      group        (N,) int8     identical assignment)
      the GT tables of C0/S0/both seeds were asserted identical in the
      earlier audits; re-asserted here against the S0 dump.

  fp_distance_seed{3407,3408}.npz
      s0only_dist  (M0,) float32 S0-only FP distance to nearest
                                 same-class GT (inf = no same-class GT
                                 in frame); S0-only = no same-class C0
                                 prediction within 2.0 m (greedy
                                 pred-to-pred, mirrors established def)
      s0only_cls   (M0,) int8
      c0only_dist / c0only_cls   symmetric C0-only FP arrays

Run:  python export_figure_data.py
"""

import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from mmengine.config import Config

ANALYSIS_DIR = Path(__file__).resolve().parents[4] / 'tools' / 'analysis'
sys.path.insert(0, str(ANALYSIS_DIR))
from evidence_stratified_analysis import (  # noqa: E402
    build_gt_table, compute_evidence, match_method)
from proposal_nms_analysis_epoch20 import pred_pred_match  # noqa: E402
from score_zero_analysis_epoch20 import strict_match_preds  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parents[1] / 'data'
OUT.mkdir(exist_ok=True)

CONFIG = ROOT / 'configs/centerpoint/mycenterpoint_nus_evidence_controlled.py'
DUMPS = {
    3407: (ROOT / 'work_dirs/evidence_controlled/analysis/C0full_dump.pkl',
           ROOT / 'work_dirs/evidence_controlled/analysis/S0full_dump.pkl'),
    3408: (ROOT / 'work_dirs/evidence_controlled_seed3408/analysis/'
           'C0full_dump.pkl',
           ROOT / 'work_dirs/evidence_controlled_seed3408/analysis/'
           'S0full_dump.pkl'),
}

GROUP_NAMES = ['Low', 'Mid', 'High']


def load_stats():
    er = Config.fromfile(str(CONFIG)).model.train_cfg.pts.evidence_radius
    return dict(mu_count=list(er['mu_count']),
                std_count=list(er['std_count']),
                mu_density=list(er['mu_density']),
                std_density=list(er['std_density']),
                beta=float(er['beta']))


def gt_arrays(dump, stats):
    rows = build_gt_table(dump)
    n = len(rows)
    cls_ids = np.array([r['cls_id'] for r in rows], dtype=np.int8)
    num_pts = np.array([r['num_pts'] for r in rows], dtype=np.int32)
    volume = np.array([r['volume'] for r in rows], dtype=np.float32)
    rng = np.array([r['lidar_range'] for r in rows], dtype=np.float32)
    evidence = np.array(
        [compute_evidence(r['num_pts'], r['volume'], r['cls_id'], stats)
         for r in rows], dtype=np.float32)
    group = np.full(n, -1, dtype=np.int8)
    for cid in range(10):
        sel = cls_ids == cid
        if sel.sum() == 0:
            continue
        q25, q75 = np.quantile(evidence[sel], [0.25, 0.75])
        group[sel & (evidence <= q25)] = 0
        group[sel & (evidence > q25) & (evidence < q75)] = 1
        group[sel & (evidence >= q75)] = 2
    return rows, cls_ids, num_pts, volume, rng, evidence, group


def main():
    stats = load_stats()

    # ---------------- E2: evidence scatter (seed-independent GT) ----
    with open(DUMPS[3407][0], 'rb') as f:
        c0_7 = pickle.load(f)
    rows, cls_ids, num_pts, volume, rng, evidence, group = \
        gt_arrays(c0_7, stats)
    np.savez_compressed(
        OUT / 'evidence_scatter.npz',
        num_pts=num_pts, volume=volume, cls_id=cls_ids,
        evidence=evidence, group=group)
    print(f'E2 evidence_scatter.npz: N={len(rows)}')

    for seed, (c0_path, s0_path) in DUMPS.items():
        with open(c0_path, 'rb') as f:
            c0 = pickle.load(f)
        with open(s0_path, 'rb') as f:
            s0 = pickle.load(f)
        rows_c, cls_c, _, _, rng_c, _, group_c = gt_arrays(c0, stats)
        rows_s, cls_s, _, _, _, _, _ = gt_arrays(s0, stats)
        assert len(rows_c) == len(rows_s)
        for a, b in zip(rows_c, rows_s):
            assert (a['token'], a['gt_idx'], a['cls_id'], a['num_pts']) \
                == (b['token'], b['gt_idx'], b['cls_id'], b['num_pts'])

        # ---------------- E1: distance recall ----------------------
        m_c0, d_c0, _ = match_method(c0, rows_c)
        m_s0, d_s0, _ = match_method(s0, rows_c)
        hit_c0 = m_c0 & (d_c0 <= 2.0)
        hit_s0 = m_s0 & (d_s0 <= 2.0)
        np.savez_compressed(
            OUT / f'distance_recall_seed{seed}.npz',
            lidar_range=rng_c, group=group_c, cls_id=cls_c,
            matched_c0=hit_c0, matched_s0=hit_s0)
        print(f'E1 distance_recall_seed{seed}.npz: N={len(rows_c)} '
              f'C0 TP={int(hit_c0.sum())} S0 TP={int(hit_s0.sum())}')

        # ---------------- E3: method-only FP distances -------------
        prec = {name: strict_match_preds(dump, rows_c, 2.0)
                for name, dump in [('C0', c0), ('S0', s0)]}
        s0_to_c0 = pred_pred_match(prec['C0'], prec['S0'])
        c0_to_s0 = pred_pred_match(prec['S0'], prec['C0'])
        only = {
            's0only': [j for j in np.where(~s0_to_c0)[0]
                       if not prec['S0'][j]['matched']],
            'c0only': [j for j in np.where(~c0_to_s0)[0]
                       if not prec['C0'][j]['matched']],
        }
        gt_xy_cls = defaultdict(list)
        for i, r in enumerate(rows_c):
            gt_xy_cls[(r['token'], r['cls_id'])].append((r['x'], r['y']))
        for k in gt_xy_cls:
            gt_xy_cls[k] = np.array(gt_xy_cls[k])

        out = {}
        for tag, idxs in only.items():
            src = 'S0' if tag == 's0only' else 'C0'
            dists = np.empty(len(idxs), dtype=np.float32)
            clss = np.empty(len(idxs), dtype=np.int8)
            for k, j in enumerate(idxs):
                rec = prec[src][j]
                same = gt_xy_cls.get((rec['token'], rec['label']))
                if same is None or len(same) == 0:
                    dists[k] = np.inf
                else:
                    dists[k] = float(np.hypot(
                        same[:, 0] - rec['x'],
                        same[:, 1] - rec['y']).min())
                clss[k] = rec['label']
            out[f'{tag}_dist'] = dists
            out[f'{tag}_cls'] = clss
            print(f'E3 seed{seed} {tag}: n={len(idxs)} '
                  f'background(inf)={int(np.isinf(dists).sum())}')
        np.savez_compressed(OUT / f'fp_distance_seed{seed}.npz', **out)


if __name__ == '__main__':
    main()
