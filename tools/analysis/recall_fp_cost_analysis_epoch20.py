"""Recall-gain vs FP-cost decomposition (epoch-20 C0 vs S0).

Question: at the fixed protocol threshold 0.1, S0 shows Recall up /
Precision down, but the threshold sweep showed the recall gain persists
while the FP cost shrinks at higher thresholds. This script maps the
full operating curve and decomposes, per threshold, where the recall
gain comes from and what the FP increase looks like.

Protocol (identical to all previous analyses): same dumps, same GT
table, same per-class evidence quantile grouping, StrictMatch@2 greedy
one-to-one class-aware matching. The only parameterized element is the
score threshold sweep requested by the task; at thr=0.1 all counts are
asserted equal to the verified audit counts.

Definitions:
  FN = GT - TP;  Recall = TP/GT;  Precision = TP/preds;
  F1 = 2PR/(P+R).
  FP cost = (FP_S0 - FP_C0) / (Recall_S0 - Recall_C0).
  S0-only TP (GT level): GT matched by S0 and not matched by C0.
  S0-only FP (prediction level): S0 prediction unmatched to GT under
      StrictMatch@2 AND with no same-class C0 prediction within 2.0 m
      (greedy pred-to-pred matching, query by score desc - same as
      proposal_nms_analysis_epoch20.py).
  Group precision keeps the established convention of
  mechanism_analysis_epoch20.py: TP_g / total preds (shared
  denominator; sums over groups to overall precision).

Reads existing dumps only; modifies nothing.
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
    FOCUS_CLASSES, SCORE_THR, build_gt_table, compute_evidence)
from evidence_stratified_matching_audit import strict_match  # noqa: E402
from proposal_nms_analysis_epoch20 import pred_pred_match  # noqa: E402

THR_SWEEP = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
             0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
GROUP_CURVE_THR = [0.10, 0.30, 0.50, 0.70]
FP_DIST_BUCKETS = [(0.0, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 10.0),
                   (10.0, np.inf)]


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
                         'recall_fp_cost_analysis_epoch20.txt')
    return p.parse_args()


def match_all(dump, gt_rows, thr, cap=2.0):
    """Verified StrictMatch algorithm at score threshold `thr`.
    Returns (records, gt_matched): records is a list of per-prediction
    dicts (token, label, score, x, y, matched, gt); gt_matched is the
    per-GT boolean array."""
    by_sc = defaultdict(list)
    for i, r in enumerate(gt_rows):
        by_sc[(r['token'], r['cls_id'])].append(i)
    records = []
    preds = defaultdict(list)
    for token, pr in dump['preds'].items():
        keep = pr['scores'] >= thr
        for box, score, label in zip(pr['boxes'][keep], pr['scores'][keep],
                                     pr['labels'][keep]):
            uid = len(records)
            records.append(dict(token=token, label=int(label),
                                score=float(score), x=float(box[0]),
                                y=float(box[1]), matched=False, gt=-1))
            preds[(token, int(label))].append(uid)
    gt_matched = np.zeros(len(gt_rows), dtype=bool)
    for key, gt_ids in by_sc.items():
        uids = preds.get(key)
        if not uids:
            continue
        uids.sort(key=lambda u: -records[u]['score'])
        gt_xy = np.array([[gt_rows[i]['x'], gt_rows[i]['y']]
                          for i in gt_ids])
        taken = np.zeros(len(gt_ids), dtype=bool)
        for u in uids:
            d = np.hypot(gt_xy[:, 0] - records[u]['x'],
                         gt_xy[:, 1] - records[u]['y'])
            d[taken] = np.inf
            j = int(np.argmin(d))
            if d[j] <= cap:
                taken[j] = True
                records[u]['matched'] = True
                records[u]['gt'] = gt_ids[j]
                gt_matched[gt_ids[j]] = True
    return records, gt_matched


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

    # nearest same-class GT lookup structures
    gt_xy_cls = {}
    for i, r in enumerate(gt_rows):
        gt_xy_cls.setdefault((r['token'], r['cls_id']), []).append(
            (r['x'], r['y']))
    for k in gt_xy_cls:
        gt_xy_cls[k] = np.array(gt_xy_cls[k])

    lines = []

    def emit(s=''):
        print(s)
        lines.append(s)

    emit('RECALL GAIN vs FP COST DECOMPOSITION (epoch-20): '
         'C0-full vs S0-full')
    emit('=' * 76)
    emit(f'input C0 dump:   {args.c0}')
    emit(f'input S0 dump:   {args.s0}')
    emit(f'analysis script: {Path(__file__).resolve()}')
    emit(f'config:          {args.config}')
    emit(f'GT: {n}; groups: Low={int((group == "Low").sum())} '
         f'Mid={int((group == "Mid").sum())} '
         f'High={int((group == "High").sum())}')
    emit('')
    emit('PROTOCOL')
    emit('  matching: StrictMatch@2 (greedy one-to-one, same class, '
         'nearest unmatched GT, preds by score desc); only the score '
         'threshold is swept as requested')
    emit('  FN = GT - TP; Recall = TP/GT; Precision = TP/preds; '
         'F1 = 2PR/(P+R)')
    emit('  FP cost = (FP_S0 - FP_C0) / (Recall_S0 - Recall_C0)')
    emit('  S0-only TP: GT matched by S0, not matched by C0 (GT level)')
    emit('  S0-only FP: S0 prediction unmatched to GT and no same-class '
         'C0 prediction within 2.0 m (pred-to-pred greedy matching, '
         'score desc)')
    emit('  group precision: TP_g / total preds (shared denominator, '
         'same convention as mechanism_analysis_epoch20.py)')
    emit('  note: dump scores are already truncated at 0.1 by test_cfg; '
         'thresholds below 0.1 cannot be evaluated')

    # ---------------- sweep ---------------------------------------
    res = {}
    for thr in THR_SWEEP:
        print(f'--- threshold {thr:.2f} ---')
        recs = {}
        gm = {}
        for name, dump in [('C0', c0), ('S0', s0)]:
            recs[name], gm[name] = match_all(dump, gt_rows, thr)
        # verification at the protocol threshold
        if abs(thr - SCORE_THR) < 1e-9:
            for name, dump in [('C0', c0), ('S0', s0)]:
                ref = int(strict_match(dump, gt_rows, 2.0)[0].sum())
                assert int(gm[name].sum()) == ref, \
                    f'{name}: thr=0.1 TP {int(gm[name].sum())} != {ref}'

        row = {}
        for name in ['C0', 'S0']:
            tp = int(gm[name].sum())
            npred = len(recs[name])
            fp = npred - tp
            row[name] = dict(preds=npred, tp=tp, fp=fp, fn=n - tp,
                             recall=tp / n,
                             prec=tp / max(npred, 1))
            row[name]['f1'] = (2 * row[name]['prec'] * row[name]['recall']
                               / max(row[name]['prec']
                                     + row[name]['recall'], 1e-12))
        # S0-only TP (GT level)
        new_tp = gm['S0'] & ~gm['C0']
        row['new_tp'] = new_tp
        row['gm_c0'] = gm['C0']
        row['gm_s0'] = gm['S0']
        # S0-only FP (prediction level)
        s0_to_c0 = pred_pred_match(recs['C0'], recs['S0'])
        s0_only_fp = np.array([
            (not r['matched']) and (not s0_to_c0[j])
            for j, r in enumerate(recs['S0'])])
        row['s0_only_fp_idx'] = np.where(s0_only_fp)[0]
        row['recs_s0'] = recs['S0']
        res[thr] = row

    # ---------------- Part 1: operating table ----------------------
    emit('\n=== PART 1. THRESHOLD OPERATING CURVE ===')
    for name in ['C0', 'S0']:
        emit(f'\n[{name}]')
        emit(f'{"thr":>6}{"preds":>9}{"TP":>8}{"FP":>9}{"FN":>8}'
             f'{"Recall":>9}{"Prec":>8}{"F1":>8}')
        for thr in THR_SWEEP:
            r = res[thr][name]
            if r['preds'] == 0:
                emit(f'{thr:>6.2f}  (no predictions)')
                continue
            emit(f'{thr:>6.2f}{r["preds"]:>9}{r["tp"]:>8}{r["fp"]:>9}'
                 f'{r["fn"]:>8}{r["recall"]:>9.4f}{r["prec"]:>8.4f}'
                 f'{r["f1"]:>8.4f}')
    emit('\n[delta S0-C0]')
    emit(f'{"thr":>6}{"dPreds":>8}{"dTP":>6}{"dFP":>8}{"dRecall":>10}'
         f'{"dPrec":>9}{"dF1":>9}')
    for thr in THR_SWEEP:
        a, b = res[thr]['C0'], res[thr]['S0']
        emit(f'{thr:>6.2f}{b["preds"] - a["preds"]:>8}'
             f'{b["tp"] - a["tp"]:>6}{b["fp"] - a["fp"]:>8}'
             f'{b["recall"] - a["recall"]:>+10.5f}'
             f'{b["prec"] - a["prec"]:>+9.4f}{b["f1"] - a["f1"]:>+9.4f}')

    # ---------------- Part 2: FP cost ------------------------------
    emit('\n=== PART 2. FP COST PER THRESHOLD ===')
    emit('FP cost = (FP_S0 - FP_C0) / (Recall_S0 - Recall_C0); '
         'also shown: dFP per +0.01 recall (same ratio x 0.01)')
    emit(f'{"thr":>6}{"dFP":>8}{"dRecall":>10}{"FPcost":>14}'
         f'{"dFP/.01rec":>12}')
    for thr in THR_SWEEP:
        a, b = res[thr]['C0'], res[thr]['S0']
        d_rec = b['recall'] - a['recall']
        d_fp = b['fp'] - a['fp']
        cost = d_fp / d_rec if abs(d_rec) > 1e-12 else float('nan')
        emit(f'{thr:>6.2f}{d_fp:>8}{d_rec:>+10.5f}{cost:>14.0f}'
             f'{cost * 0.01:>12.1f}')

    # ---------------- Part 3: recall gain source -------------------
    emit('\n=== PART 3. S0-ONLY TP SOURCE (GT matched by S0, not C0) ===')
    emit('per threshold: count, evidence-group split, class split')
    for thr in THR_SWEEP:
        nt = res[thr]['new_tp']
        cnt = int(nt.sum())
        emit(f'\nthr={thr:.2f}  S0-only TP (newly matched GT): {cnt}')
        emit('  group: ' + ', '.join(
            f'{g}={int((nt & (group == g)).sum())}'
            f'({(nt & (group == g)).sum() / max(cnt, 1):.3f})'
            for g in ['Low', 'Mid', 'High']))
        emit('  class: ' + ', '.join(
            f'{cn}={int((nt & (cls_ids == class_names.index(cn))).sum())}'
            for cn in FOCUS_CLASSES))

    # ---------------- Part 4: FP characterization ------------------
    emit('\n=== PART 4. S0-ONLY FP DISTANCE CHARACTERIZATION ===')
    emit('distance to nearest same-class GT; "background" = no '
         'same-class GT in the frame')
    emit(f'{"thr":>6}{"nFP":>8}{"0-1m":>7}{"1-2m":>7}{"2-5m":>7}'
         f'{"5-10m":>7}{">10m":>8}{"bkgrnd":>8}')
    for thr in THR_SWEEP:
        idx = res[thr]['s0_only_fp_idx']
        recs = res[thr]['recs_s0']
        # group by (token, label) for vectorized distance computation
        by_key = defaultdict(list)
        for j in idx:
            by_key[(recs[j]['token'], recs[j]['label'])].append(j)
        bucket_cnt = np.zeros(len(FP_DIST_BUCKETS), dtype=int)
        n_bg = 0
        for key, js in by_key.items():
            gxy = gt_xy_cls.get(key)
            if gxy is None:
                n_bg += len(js)
                continue
            pxy = np.array([[recs[j]['x'], recs[j]['y']] for j in js])
            dmin = np.hypot(pxy[:, 0:1] - gxy[:, 0],
                            pxy[:, 1:2] - gxy[:, 1]).min(axis=1)
            for bi, (lo, hi) in enumerate(FP_DIST_BUCKETS):
                bucket_cnt[bi] += int(((dmin >= lo) & (dmin < hi)).sum())
        emit(f'{thr:>6.2f}{len(idx):>8}'
             + ''.join(f'{c:>7}' for c in bucket_cnt[:4])
             + f'{bucket_cnt[4]:>8}{n_bg:>8}')

    # ---------------- Part 5: optimal operating points -------------
    emit('\n=== PART 5. OPTIMAL OPERATING POINTS ===')
    for name in ['C0', 'S0']:
        best = max(THR_SWEEP, key=lambda t: res[t][name]['f1'])
        r = res[best][name]
        emit(f'{name} best F1: thr={best:.2f} recall={r["recall"]:.4f} '
             f'precision={r["prec"]:.4f} F1={r["f1"]:.4f}')
    rec_ref = res[SCORE_THR]['C0']['recall']
    feas = [t for t in THR_SWEEP if res[t]['S0']['recall'] >= rec_ref]
    emit(f'\nrecall constraint: Recall_S0 >= Recall_C0@0.1 '
         f'({rec_ref:.4f})')
    if feas:
        best = max(feas, key=lambda t: res[t]['S0']['prec'])
        r = res[best]['S0']
        a = res[SCORE_THR]['C0']
        emit(f'S0 highest-precision feasible point: thr={best:.2f} '
             f'precision={r["prec"]:.4f} recall={r["recall"]:.4f}')
        emit(f'vs C0@0.1: dRecall={r["recall"] - a["recall"]:+.5f} '
             f'dPrecision={r["prec"] - a["prec"]:+.4f}')
        emit(f'feasible thresholds: {[f"{t:.2f}" for t in feas]}')
    else:
        emit('no threshold in the sweep satisfies the constraint')

    # ---------------- Part 6: group-wise curve ---------------------
    emit('\n=== PART 6. GROUP-WISE OPERATING CURVE ===')
    emit('Recall_g = TP_g/GT_g; Precision_g = TP_g/total preds '
         '(shared denominator); F1 from these two')
    for thr in GROUP_CURVE_THR:
        emit(f'\nthr={thr:.2f}')
        emit(f'{"group":<6}{"m":<4}{"Recall":>9}{"Prec":>9}{"F1":>9}')
        for gname in ['Low', 'Mid', 'High']:
            gsel = group == gname
            ng = int(gsel.sum())
            for name in ['C0', 'S0']:
                gm_name = res[thr]['gm_c0' if name == 'C0'
                                   else 'gm_s0']
                tp_g = int((gm_name & gsel).sum())
                prec_g = tp_g / res[thr][name]['preds']
                rec_g = tp_g / ng
                f1_g = 2 * prec_g * rec_g / max(prec_g + rec_g, 1e-12)
                emit(f'{gname:<6}{name:<4}{rec_g:>9.4f}{prec_g:>9.4f}'
                     f'{f1_g:>9.4f}')

    # ---------------- limitations + summary ------------------------
    emit('\n=== LIMITATIONS ===')
    emit('  - dump scores are pre-truncated at 0.1 by test_cfg; the '
         'operating curve below 0.1 cannot be evaluated.')
    emit('  - group precision uses the shared-denominator convention; '
         'predictions carry no evidence group.')
    emit('  - S0-only definitions (TP at GT level, FP at prediction '
         'level) are correspondence measures against C0 outputs, not '
         'network-internal provenance.')
    emit('  - FP cost ratio is unstable when the recall delta is '
         'close to zero; values are reported as computed.')

    emit('\n=== FACTUAL SUMMARY (statistics only; no causal claims) ===')
    for name in ['C0', 'S0']:
        best = max(THR_SWEEP, key=lambda t: res[t][name]['f1'])
        r = res[best][name]
        emit(f'  - Observed {name} best F1 at thr={best:.2f} '
             f'(recall={r["recall"]:.4f}, precision={r["prec"]:.4f}, '
             f'F1={r["f1"]:.4f}).')
    a, b = res[SCORE_THR]['C0'], res[SCORE_THR]['S0']
    emit(f'  - Observed at the protocol point thr=0.1: dRecall='
         f'{b["recall"] - a["recall"]:+.5f}, dPrecision='
         f'{b["prec"] - a["prec"]:+.4f}, dF1={b["f1"] - a["f1"]:+.4f}.')
    emit('  - Statistics show the per-threshold FP cost and the '
         'S0-only TP group/class sources in Parts 2-3; the distance '
         'characterization of S0-only FPs per threshold is reported '
         'in Part 4.')
    if feas:
        emit(f'  - Results indicate the recall-constrained '
             f'(>= C0@0.1) highest-precision S0 point is reported in '
             f'Part 5 with its deltas.')
    emit('  - Observed group-wise operating points at '
         'thr in {0.1,0.3,0.5,0.7} are reported in Part 6.')

    with open(args.out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nreport saved to {args.out}')


if __name__ == '__main__':
    main()
