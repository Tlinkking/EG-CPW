"""Proposal-level NMS before/after analysis (epoch-20 C0 vs S0).

Question: S0 vs C0 shows Recall up, Precision down, +56,067 final
predictions (score>=0.1). Do the extra FPs come from (A) more
low-quality pre-NMS proposals or (B) insufficient NMS suppression?

Dump capability check (both dumps): top-level keys are preds / gts /
class_names / task_class_names / pc_range / voxel_size /
out_size_factor / checkpoint; per-token predictions hold only
post-NMS boxes/scores/labels. No pre-NMS candidates, heatmap peaks or
top-k fields exist, so Parts 1-2 (pre/post-NMS proposal counts and
suppression behavior) cannot be computed offline:
    "NMS-before information unavailable from current dump"
No checkpoint is reloaded and no inference is re-run.

Executable parts (3-6) keep the established protocol: score threshold
0.1 (except the requested sensitivity sweep), StrictMatch@2 greedy
one-to-one class-aware matching, identical GT/evidence grouping.

S0-only prediction definition (mirrors the established matching
logic): per frame per class, S0 predictions are processed in
descending score order and greedily matched to the nearest unmatched
C0 prediction within 2.0 m BEV distance; S0 predictions left unmatched
are "S0-only". C0-only is computed symmetrically.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pickle
from mmengine.config import Config

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence_stratified_analysis import (  # noqa: E402
    BOOT_SEED, FOCUS_CLASSES, LOW_SUPPORT, N_BOOT, SCORE_THR, TAUS,
    build_gt_table, compute_evidence)
from evidence_stratified_matching_audit import strict_match  # noqa: E402
from score_zero_analysis_epoch20 import (  # noqa: E402
    strict_match_preds)

THR_SWEEP = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
FP_DIST_BUCKETS = [(0.0, 1.0), (1.0, 2.0), (2.0, 5.0),
                   (5.0, np.inf)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        '--config',
        default='configs/centerpoint/mycenterpoint_nus_evidence_controlled.py')
    p.add_argument(
        '--c0', default='work_dirs/evidence_controlled/analysis/'
                        'C0full_dump.pkl')
    p.add_argument(
        '--s0', default='work_dirs/evidence_controlled/analysis/'
                        'S0full_dump.pkl')
    p.add_argument(
        '--out', default='work_dirs/evidence_controlled/analysis/'
                         'proposal_nms_analysis_epoch20.txt')
    return p.parse_args()


def strict_match_thr(dump, gt_rows, thr, cap=2.0):
    """Verified StrictMatch algorithm with a parameterized score
    threshold (used only for the requested sensitivity sweep)."""
    n = len(gt_rows)
    matched = np.zeros(n, dtype=bool)
    by_sc = defaultdict(list)
    for i, r in enumerate(gt_rows):
        by_sc[(r['token'], r['cls_id'])].append(i)
    preds = defaultdict(list)
    for token, pr in dump['preds'].items():
        keep = pr['scores'] >= thr
        for box, score, label in zip(pr['boxes'][keep], pr['scores'][keep],
                                     pr['labels'][keep]):
            preds[(token, int(label))].append(
                (float(score), float(box[0]), float(box[1])))
    n_pred = sum(len(v) for v in preds.values())
    for key, gt_ids in by_sc.items():
        ps = preds.get(key)
        if not ps:
            continue
        ps.sort(key=lambda p: -p[0])
        gt_xy = np.array([[gt_rows[i]['x'], gt_rows[i]['y']]
                          for i in gt_ids])
        taken = np.zeros(len(gt_ids), dtype=bool)
        for score, px, py in ps:
            d = np.hypot(gt_xy[:, 0] - px, gt_xy[:, 1] - py)
            d[taken] = np.inf
            j = int(np.argmin(d))
            if d[j] <= cap:
                taken[j] = True
                matched[gt_ids[j]] = True
    return matched, n_pred


def pred_pred_match(recs_ref, recs_qry, cap=2.0):
    """Greedy one-to-one matching of query predictions to reference
    predictions (query processed by descending score, nearest unmatched
    reference within `cap`, same frame and class). Mirrors the
    established StrictMatch logic with predictions as targets."""
    by_ref = defaultdict(list)
    by_qry = defaultdict(list)
    for i, r in enumerate(recs_ref):
        by_ref[(r['token'], r['label'])].append(i)
    for j, r in enumerate(recs_qry):
        by_qry[(r['token'], r['label'])].append(j)
    qry_matched = np.zeros(len(recs_qry), dtype=bool)
    for key, q_ids in by_qry.items():
        r_ids = by_ref.get(key)
        if not r_ids:
            continue
        q_ids.sort(key=lambda j: -recs_qry[j]['score'])
        r_xy = np.array([[recs_ref[i]['x'], recs_ref[i]['y']]
                         for i in r_ids])
        taken = np.zeros(len(r_ids), dtype=bool)
        for j in q_ids:
            d = np.hypot(r_xy[:, 0] - recs_qry[j]['x'],
                         r_xy[:, 1] - recs_qry[j]['y'])
            d[taken] = np.inf
            k = int(np.argmin(d))
            if d[k] <= cap:
                taken[k] = True
                qry_matched[j] = True
    return qry_matched


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    er = cfg.model.train_cfg.pts.evidence_radius
    stats = dict(
        mu_count=list(er['mu_count']), std_count=list(er['std_count']),
        mu_density=list(er['mu_density']),
        std_density=list(er['std_density']), beta=float(er['beta']))

    with open(args.c0, 'rb') as f:
        c0 = pickle.load(f)
    with open(args.s0, 'rb') as f:
        s0 = pickle.load(f)
    class_names = c0['class_names']

    gt_rows = build_gt_table(c0)
    gt_s0 = build_gt_table(s0)
    assert len(gt_rows) == len(gt_s0)
    for a, b in zip(gt_rows, gt_s0):
        assert (a['token'], a['gt_idx'], a['cls_id'], a['num_pts']) == \
               (b['token'], b['gt_idx'], b['cls_id'], b['num_pts'])
    n = len(gt_rows)

    cls_ids = np.array([r['cls_id'] for r in gt_rows])
    evidence = np.array([
        compute_evidence(r['num_pts'], r['volume'], r['cls_id'], stats)
        for r in gt_rows])
    group = np.full(n, '', dtype=object)
    for cid in range(len(class_names)):
        sel = cls_ids == cid
        if sel.sum() == 0:
            continue
        q25, q75 = np.quantile(evidence[sel], [0.25, 0.75])
        group[sel & (evidence <= q25)] = 'Low'
        group[sel & (evidence > q25) & (evidence < q75)] = 'Mid'
        group[sel & (evidence >= q75)] = 'High'

    lines = []

    def emit(s=''):
        print(s)
        lines.append(s)

    emit('PROPOSAL-LEVEL NMS BEFORE/AFTER ANALYSIS (epoch-20): '
         'C0-full vs S0-full')
    emit('=' * 76)
    emit(f'input C0 dump:   {args.c0}')
    emit(f'input S0 dump:   {args.s0}')
    emit(f'analysis script: {Path(__file__).resolve()}')
    emit(f'config:          {args.config}')

    # ---------------- Step 0: dump capability ----------------------
    emit('\n=== STEP 0. DUMP FIELD CHECK ===')
    emit(f'top-level keys: {sorted(c0.keys())}')
    tok = next(iter(c0['preds']))
    emit(f'preds[token] keys: {sorted(c0["preds"][tok].keys())}')
    emit('no pre-NMS / heatmap-peak / top-k candidate fields found')
    emit('NMS-before information unavailable from current dump')

    # ---------------- Part 1/2: unavailable ------------------------
    emit('\n=== PART 1. PROPOSAL COUNT BEFORE/AFTER NMS ===')
    emit('NMS-before information unavailable from current dump; '
         'pre-NMS candidate/TP/FP counts and retention ratios cannot '
         'be computed offline. No checkpoint reload or re-inference '
         'was performed.')
    emit('\n=== PART 2. NMS SUPPRESSION BEHAVIOR ===')
    emit('NMS-before information unavailable from current dump; '
         'suppression/removal statistics cannot be computed. '
         'Whether the extra FPs originate at proposal generation (A) '
         'or NMS suppression (B) cannot be distinguished from these '
         'dumps.')

    # ---------------- Part 3: threshold sensitivity ----------------
    emit('\n=== PART 3. SCORE THRESHOLD SENSITIVITY (final predictions,'
         ' StrictMatch@2) ===')
    emit(f'{"thr":>6}{"m":<4}{"preds":>9}{"TP":>8}{"FP":>9}'
         f'{"prec":>8}{"recall":>8}')
    sweep = {}
    for thr in THR_SWEEP:
        for name, dump in [('C0', c0), ('S0', s0)]:
            mt, n_pred = strict_match_thr(dump, gt_rows, thr)
            tp = int(mt.sum())
            sweep[(name, thr)] = (n_pred, tp)
            emit(f'{thr:>6.2f}{name:<4}{n_pred:>9}{tp:>8}'
                 f'{n_pred - tp:>9}{tp / max(n_pred, 1):>8.4f}'
                 f'{tp / n:>8.4f}')
    # verification at the protocol threshold
    for name, dump in [('C0', c0), ('S0', s0)]:
        ref = int(strict_match(dump, gt_rows, 2.0)[0].sum())
        got = sweep[(name, 0.1)][1]
        assert ref == got, f'{name}: thr=0.1 TP {got} != audit {ref}'
    emit('verification: TP at thr=0.1 equals the audit StrictMatch@2 '
         'counts (C0=137322, S0=137984)')
    emit('\ndelta (S0-C0) per threshold:')
    emit(f'{"thr":>6}{"dPreds":>9}{"dTP":>7}{"dFP":>8}{"dPrec":>9}'
         f'{"dRecall":>9}')
    for thr in THR_SWEEP:
        pc, tc = sweep[('C0', thr)]
        ps, ts = sweep[('S0', thr)]
        emit(f'{thr:>6.2f}{ps - pc:>9}{ts - tc:>7}'
             f'{(ps - ts) - (pc - tc):>8}'
             f'{ts / max(ps, 1) - tc / max(pc, 1):>+9.4f}'
             f'{ts / n - tc / n:>+9.5f}')

    # ---------------- Part 4: S0-only predictions ------------------
    emit('\n=== PART 4. S0-ONLY PREDICTION SPATIAL ANALYSIS ===')
    emit('S0-only: S0 prediction (score>=0.1) with no same-class C0 '
         'prediction within 2.0 m (greedy pred-to-pred matching, '
         'query by score desc - mirrors StrictMatch logic)')
    prec = {name: strict_match_preds(dump, gt_rows, 2.0)
            for name, dump in [('C0', c0), ('S0', s0)]}
    for name in ['C0', 'S0']:
        n_tp = sum(r['matched'] for r in prec[name])
        assert n_tp == sweep[(name, 0.1)][1]

    s0_to_c0 = pred_pred_match(prec['C0'], prec['S0'])
    c0_to_s0 = pred_pred_match(prec['S0'], prec['C0'])
    s0_only = ~s0_to_c0
    c0_only = ~c0_to_s0
    emit(f'C0 preds={len(prec["C0"])}, S0 preds={len(prec["S0"])}')
    emit(f'S0-only predictions: {int(s0_only.sum())} '
         f'({s0_only.mean():.4f} of S0); C0-only predictions: '
         f'{int(c0_only.sum())} ({c0_only.mean():.4f} of C0)')

    gt_matched_c0 = strict_match(c0, gt_rows, 2.0)[0]
    only_idx = np.where(s0_only)[0]
    only_tp = [j for j in only_idx if prec['S0'][j]['matched']]
    only_fp = [j for j in only_idx if not prec['S0'][j]['matched']]
    emit(f'\nS0-only breakdown: TP gain={len(only_tp)}, '
         f'FP increase={len(only_fp)}')
    new_gt = sum(1 for j in only_tp
                 if not gt_matched_c0[prec['S0'][j]['gt']])
    emit(f'  of the S0-only TP: matched GT newly detected by S0 '
         f'(GT not matched by C0): {new_gt}; matched GT already '
         f'detected by C0 (replacement/duplicate TP): '
         f'{len(only_tp) - new_gt}')

    # nearest-GT distance of S0-only FPs
    gt_xy_cls = defaultdict(list)
    gt_xy_any = defaultdict(list)
    for i, r in enumerate(gt_rows):
        gt_xy_cls[(r['token'], r['cls_id'])].append((r['x'], r['y']))
        gt_xy_any[r['token']].append((r['x'], r['y']))
    for k in gt_xy_cls:
        gt_xy_cls[k] = np.array(gt_xy_cls[k])
    for k in gt_xy_any:
        gt_xy_any[k] = np.array(gt_xy_any[k])

    def nearest_d(rec):
        same = gt_xy_cls.get((rec['token'], rec['label']))
        anyg = gt_xy_any.get(rec['token'])
        d_same = float(np.hypot(same[:, 0] - rec['x'],
                                same[:, 1] - rec['y']).min()) \
            if same is not None and len(same) else np.inf
        d_any = float(np.hypot(anyg[:, 0] - rec['x'],
                               anyg[:, 1] - rec['y']).min()) \
            if anyg is not None and len(anyg) else np.inf
        return d_same, d_any

    emit(f'\nS0-only FP (n={len(only_fp)}): distance to nearest GT')
    emit(f'{"bucket":<8}{"sameClass":>10}{"anyClass":>10}')
    for lo, hi in FP_DIST_BUCKETS:
        c_same = c_any = 0
        for j in only_fp:
            d_same, d_any = nearest_d(prec['S0'][j])
            c_same += int(lo <= d_same < hi)
            c_any += int(lo <= d_any < hi)
        hi_s = 'inf' if np.isinf(hi) else f'{hi:.0f}'
        emit(f'[{lo:.0f},{hi_s}){"":<3}{c_same:>10}{c_any:>10}')
    d_same_all = np.array([nearest_d(prec['S0'][j])[0] for j in only_fp])
    d_same_all = d_same_all[np.isfinite(d_same_all)]
    emit(f'nearest same-class GT distance: mean='
         f'{d_same_all.mean():.3f} median={np.median(d_same_all):.3f} '
         f'(over {len(d_same_all)} FPs with a same-class GT in frame)')

    # ---------------- Part 5: class-wise FP increase ---------------
    emit('\n=== PART 5. CLASS-WISE S0-ONLY FP INCREASE ===')
    emit(f'{"class":<22}{"S0onlyFP":>9}{"pct":>8}{"S0onlyTP":>9}')
    per_cls_fp = defaultdict(int)
    per_cls_tp = defaultdict(int)
    for j in only_fp:
        per_cls_fp[prec['S0'][j]['label']] += 1
    for j in only_tp:
        per_cls_tp[prec['S0'][j]['label']] += 1
    for cid in sorted(set(list(per_cls_fp) + list(per_cls_tp))):
        emit(f'{class_names[cid]:<22}{per_cls_fp[cid]:>9}'
             f'{per_cls_fp[cid] / max(len(only_fp), 1):>8.4f}'
             f'{per_cls_tp[cid]:>9}')

    # ---------------- Part 6: evidence of new TP -------------------
    emit('\n=== PART 6. EVIDENCE GROUP OF S0-ONLY TP (matched GT) ===')
    emit(f'{"group":<6}{"allS0onlyTP":>12}{"newlyDet":>9}')
    for gname in ['Low', 'Mid', 'High']:
        g_tp = sum(1 for j in only_tp
                   if group[prec['S0'][j]['gt']] == gname)
        g_new = sum(1 for j in only_tp
                    if group[prec['S0'][j]['gt']] == gname
                    and not gt_matched_c0[prec['S0'][j]['gt']])
        emit(f'{gname:<6}{g_tp:>12}{g_new:>9}')
    emit('S0-only FP carry no per-instance evidence (predictions have '
         'no GT); no evidence grouping is applied to them.')

    # ---------------- limitations + summary ------------------------
    emit('\n=== LIMITATIONS ===')
    emit('  - NMS-before information unavailable from current dump; '
         'proposal-generation vs NMS-suppression origins (question '
         'A vs B) cannot be separated offline.')
    emit('  - S0-only is defined via greedy pred-to-pred matching '
         '(2.0 m, same class, score desc); it identifies predictions '
         'without a close C0 counterpart, not a provenance inside the '
         'network.')
    emit('  - threshold sensitivity uses final (post-NMS) predictions '
         'only.')

    emit('\n=== FACTUAL SUMMARY (statistics only; no causal claims) ===')
    pcnt_c, tp_c = sweep[('C0', 0.1)]
    pcnt_s, tp_s = sweep[('S0', 0.1)]
    emit(f'  - NMS-before information unavailable from current dump; '
         f'Parts 1-2 not computable offline.')
    emit(f'  - Observed at thr=0.1: preds {pcnt_c}->{pcnt_s}, '
         f'TP {tp_c}->{tp_s}, FP {pcnt_c - tp_c}->{pcnt_s - tp_s}.')
    emit(f'  - Observed threshold sweep (Part 3): the precision delta '
         f'at thr=0.5 is '
         f'{sweep[("S0", 0.5)][1] / max(sweep[("S0", 0.5)][0], 1) - sweep[("C0", 0.5)][1] / max(sweep[("C0", 0.5)][0], 1):+.4f} '
         f'vs '
         f'{sweep[("S0", 0.1)][1] / max(sweep[("S0", 0.1)][0], 1) - sweep[("C0", 0.1)][1] / max(sweep[("C0", 0.1)][0], 1):+.4f} '
         f'at thr=0.1.')
    emit(f'  - Observed S0-only predictions: {int(s0_only.sum())} '
         f'(TP gain {len(only_tp)}, of which {new_gt} on GTs not '
         f'detected by C0; FP increase {len(only_fp)}); C0-only '
         f'predictions: {int(c0_only.sum())}.')
    top_cls = max(per_cls_fp, key=per_cls_fp.get)
    emit(f'  - Statistics show the largest S0-only FP class is '
         f'{class_names[top_cls]} ({per_cls_fp[top_cls]}, '
         f'{per_cls_fp[top_cls] / max(len(only_fp), 1):.4f} of '
         f'S0-only FPs).')

    with open(args.out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nreport saved to {args.out}')


if __name__ == '__main__':
    main()
