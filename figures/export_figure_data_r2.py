"""Export round-2 figure data (reviewer-revision set).

Reads the existing epoch-20 dumps only; no inference, no new metrics.
Same offline protocol as export_figure_data.py (score>=0.1,
StrictMatch@2 greedy one-to-one class-aware matching, per-class
evidence 25/75 quantile groups on the fixed val GT table).

Adds what the revised Figures 3/5/7 need:

  distance_recall_seed{3407,3408}.npz   (regenerated, +token)
      + token  (N,) U32   frame token per GT (frame-cluster bootstrap)
      existing arrays re-computed and asserted identical to round 1

  sweep_matched_seed{3407,3408}.npz     (new)
      thresholds   (15,) float32  0.10..0.80 step 0.05
      token        (N,) U32
      group/cls_id/lidar_range  (N,)
      matched_c0 / matched_s0   (N,15) bool  StrictMatch@2 per threshold

  gained_tp_seed{3407,3408}.npz         (new)
      per GT with matched_s0 & ~matched_c0 (thr=0.1):
      s0_score     (K,) float32  score of the S0 prediction that matched
      s0_dist      (K,) float32  matched distance (<=2.0 by construction)
      cls_id / group / num_pts / lidar_range / token

  fp_distance_seed{3407,3408}.npz       (regenerated, +score fields)
      + s0only_score / c0only_score  score of the method-only FP

Run:  python export_figure_data_r2.py
"""

import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from mmengine.config import Config

ANALYSIS_DIR = Path(__file__).resolve().parents[4] / 'tools' / 'analysis'
sys.path.insert(0, str(ANALYSIS_DIR))
import evidence_stratified_analysis as esa  # noqa: E402
from evidence_stratified_analysis import (  # noqa: E402
    build_gt_table, compute_evidence, match_method)
from proposal_nms_analysis_epoch20 import pred_pred_match  # noqa: E402
from score_zero_analysis_epoch20 import strict_match_preds  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parents[1] / 'data'

CONFIG = ROOT / 'configs/centerpoint/mycenterpoint_nus_evidence_controlled.py'
DUMPS = {
    3407: (ROOT / 'work_dirs/evidence_controlled/analysis/C0full_dump.pkl',
           ROOT / 'work_dirs/evidence_controlled/analysis/S0full_dump.pkl'),
    3408: (ROOT / 'work_dirs/evidence_controlled_seed3408/analysis/'
           'C0full_dump.pkl',
           ROOT / 'work_dirs/evidence_controlled_seed3408/analysis/'
           'S0full_dump.pkl'),
}
SWEEP_THR = np.round(np.arange(0.10, 0.801, 0.05), 2).astype(np.float32)


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
    token = np.array([r['token'] for r in rows], dtype='U32')
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
    return rows, cls_ids, num_pts, volume, rng, token, evidence, group


def main():
    stats = load_stats()

    for seed, (c0_path, s0_path) in DUMPS.items():
        with open(c0_path, 'rb') as f:
            c0 = pickle.load(f)
        with open(s0_path, 'rb') as f:
            s0 = pickle.load(f)
        rows_c, cls_c, num_pts, _, rng_c, tok_c, _, group_c = \
            gt_arrays(c0, stats)
        rows_s = build_gt_table(s0)
        assert len(rows_c) == len(rows_s)
        for a, b in zip(rows_c, rows_s):
            assert (a['token'], a['gt_idx'], a['cls_id'], a['num_pts']) \
                == (b['token'], b['gt_idx'], b['cls_id'], b['num_pts'])

        # ---- E1r: distance recall + token (assert identical to r1) --
        esa.SCORE_THR = 0.1
        m_c0, d_c0, _ = match_method(c0, rows_c)
        m_s0, d_s0, _ = match_method(s0, rows_c)
        hit_c0 = m_c0 & (d_c0 <= 2.0)
        hit_s0 = m_s0 & (d_s0 <= 2.0)
        old = np.load(OUT / f'distance_recall_seed{seed}.npz')
        assert np.array_equal(old['matched_c0'], hit_c0)
        assert np.array_equal(old['matched_s0'], hit_s0)
        np.savez_compressed(
            OUT / f'distance_recall_seed{seed}.npz',
            lidar_range=rng_c, group=group_c, cls_id=cls_c,
            matched_c0=hit_c0, matched_s0=hit_s0, token=tok_c)
        print(f'E1r seed{seed}: N={len(rows_c)} C0 TP={int(hit_c0.sum())} '
              f'S0 TP={int(hit_s0.sum())} (r1 arrays asserted identical)')

        # ---- E4: threshold sweep matched flags ----------------------
        n = len(rows_c)
        sw_c0 = np.zeros((n, len(SWEEP_THR)), dtype=bool)
        sw_s0 = np.zeros((n, len(SWEEP_THR)), dtype=bool)
        for ti, thr in enumerate(SWEEP_THR):
            esa.SCORE_THR = float(thr)
            mc, dc, _ = match_method(c0, rows_c)
            ms, ds, _ = match_method(s0, rows_c)
            sw_c0[:, ti] = mc & (dc <= 2.0)
            sw_s0[:, ti] = ms & (ds <= 2.0)
            print(f'E4 seed{seed} thr={thr:.2f} '
                  f'C0={int(sw_c0[:, ti].sum())} '
                  f'S0={int(sw_s0[:, ti].sum())}')
        esa.SCORE_THR = 0.1
        # cross-check sweep thr=0.10 column against protocol matching
        assert np.array_equal(sw_c0[:, 0], hit_c0)
        assert np.array_equal(sw_s0[:, 0], hit_s0)
        np.savez_compressed(
            OUT / f'sweep_matched_seed{seed}.npz',
            thresholds=SWEEP_THR, token=tok_c, group=group_c,
            cls_id=cls_c, lidar_range=rng_c,
            matched_c0=sw_c0, matched_s0=sw_s0)

        # ---- E5: gained TP (S0 matched, C0 unmatched) ---------------
        prec_s0 = strict_match_preds(s0, rows_c, 2.0)
        gt2pred = {}
        for rec in prec_s0:
            if rec['matched']:
                gt2pred[rec['gt']] = rec
        gained = np.where(hit_s0 & ~hit_c0)[0]
        K = len(gained)
        g_score = np.empty(K, dtype=np.float32)
        g_dist = np.empty(K, dtype=np.float32)
        for k, gi in enumerate(gained):
            rec = gt2pred[gi]
            g_score[k] = rec['score']
            g_dist[k] = np.hypot(rec['x'] - rows_c[gi]['x'],
                                 rec['y'] - rows_c[gi]['y'])
        np.savez_compressed(
            OUT / f'gained_tp_seed{seed}.npz',
            s0_score=g_score, s0_dist=g_dist,
            cls_id=cls_c[gained], group=group_c[gained],
            num_pts=num_pts[gained], lidar_range=rng_c[gained],
            token=tok_c[gained])
        print(f'E5 seed{seed}: gained TP K={K} '
              f'(zero-pt={int((num_pts[gained] == 0).sum())})')

        # ---- E3r: method-only FP + scores ---------------------------
        prec = {'C0': strict_match_preds(c0, rows_c, 2.0)}
        prec['S0'] = prec_s0
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

        old_fp = np.load(OUT / f'fp_distance_seed{seed}.npz')
        out = {}
        for tag, idxs in only.items():
            src = 'S0' if tag == 's0only' else 'C0'
            dists = np.empty(len(idxs), dtype=np.float32)
            clss = np.empty(len(idxs), dtype=np.int8)
            scrs = np.empty(len(idxs), dtype=np.float32)
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
                scrs[k] = rec['score']
            out[f'{tag}_dist'] = dists
            out[f'{tag}_cls'] = clss
            out[f'{tag}_score'] = scrs
            assert np.array_equal(old_fp[f'{tag}_cls'], clss)
            print(f'E3r seed{seed} {tag}: n={len(idxs)} '
                  f'background(inf)={int(np.isinf(dists).sum())} '
                  f'(r1 cls asserted identical)')
        np.savez_compressed(OUT / f'fp_distance_seed{seed}.npz', **out)


if __name__ == '__main__':
    main()
