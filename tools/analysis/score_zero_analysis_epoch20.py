"""Score calibration + zero-point GT source analysis (epoch-20 C0 vs S0).

Follow-up to mechanism_analysis_epoch20.py. Motivation:
  S0 vs C0 shows Recall up, Precision down, localization slightly worse;
  the Low-group StrictMatch@2 gain includes zero-point GTs, which never
  entered weighted positive supervision (filter_empty_gt). This script
  checks (1) whether the precision drop is a score-calibration effect
  (more low-quality predictions) and (2) what input context surrounds
  the improved zero-point GTs.

Protocol identical to the audit: same dumps, same GT table, same
per-class evidence quantile grouping, score threshold 0.1,
StrictMatch@2 greedy one-to-one class-aware matching. Nothing here
modifies any existing file.

Prediction-side TP/FP: the same greedy matching is executed recording
the prediction side; a prediction is TP iff it is selected by the
2.0 m assignment. TP counts are asserted equal to the GT-side matched
counts of the verified audit matching.

Dump content note: the dumps contain per-sample-token predictions and
GT rows only. They carry no scene id, timestamp or sweep linkage, so
scene-level and multi-frame visibility analyses are unavailable.
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
    BOOT_SEED, LOW_SUPPORT, N_BOOT, SCORE_THR, TAUS,
    build_gt_table, compute_evidence)
from evidence_stratified_matching_audit import strict_match  # noqa: E402

SCORE_BINS = [(0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6),
              (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0 + 1e-9)]
NB_RADII = [1.0, 2.0, 5.0]


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
                         'score_zero_analysis_epoch20.txt')
    return p.parse_args()


def strict_match_preds(dump, gt_rows, cap):
    """Prediction-side record of the verified StrictMatch greedy
    one-to-one assignment (identical algorithm, distance cap `cap`)."""
    by_sc = defaultdict(list)
    for i, r in enumerate(gt_rows):
        by_sc[(r['token'], r['cls_id'])].append(i)

    records = []
    preds = defaultdict(list)
    for token, pr in dump['preds'].items():
        keep = pr['scores'] >= SCORE_THR
        for box, score, label in zip(pr['boxes'][keep], pr['scores'][keep],
                                     pr['labels'][keep]):
            uid = len(records)
            records.append(dict(token=token, label=int(label),
                                score=float(score), x=float(box[0]),
                                y=float(box[1]), matched=False, gt=-1))
            preds[(token, int(label))].append(uid)

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
    return records


def dist_stats(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return {k: float('nan') for k in
                ['mean', 'std', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95',
                 'p99']}
    return dict(mean=float(x.mean()), std=float(x.std()),
                p10=float(np.percentile(x, 10)),
                p25=float(np.percentile(x, 25)),
                p50=float(np.percentile(x, 50)),
                p75=float(np.percentile(x, 75)),
                p90=float(np.percentile(x, 90)),
                p95=float(np.percentile(x, 95)),
                p99=float(np.percentile(x, 99)))


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
    tokens = np.array([r['token'] for r in gt_rows])
    num_pts = np.array([r['num_pts'] for r in gt_rows])
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

    # GT-side verified matching (StrictMatch@tau) + prediction-side
    strict = {}
    prec = {}
    for name, dump in [('C0', c0), ('S0', s0)]:
        for tau in TAUS:
            strict[(name, tau)] = strict_match(dump, gt_rows, tau)
        prec[name] = strict_match_preds(dump, gt_rows, 2.0)
        n_tp = sum(r['matched'] for r in prec[name])
        n_gt_m = int(strict[(name, 2.0)][0].sum())
        assert n_tp == n_gt_m, \
            f'{name}: prediction-side TP {n_tp} != GT-side {n_gt_m}'

    lines = []

    def emit(s=''):
        print(s)
        lines.append(s)

    emit('SCORE CALIBRATION + ZERO-POINT SOURCE ANALYSIS (epoch-20): '
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
    emit(f'  score threshold: {SCORE_THR}; matching: StrictMatch@2 '
         '(greedy one-to-one, same class, nearest unmatched GT, '
         'preds by score desc) - identical to the matching audit')
    emit('  TP: prediction selected by the 2m assignment; FP: prediction '
         'with score>=threshold not selected')
    emit('  evidence grouping: per-class quantiles on the fixed val GT '
         'table (identical assignment for C0/S0)')
    emit('  recall contribution of bin b = TP_b / total GT')

    # ---------------- Part 1.1 overall score distribution ----------
    emit('\n' + '=' * 76)
    emit('PART 1.1 OVERALL SCORE DISTRIBUTION (predictions, score>=0.1)')
    keys = ['mean', 'std', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95',
            'p99']
    emit(f'{"m":<4}{"count":>9}' + ''.join(f'{k:>9}' for k in keys))
    all_scores = {}
    for name in ['C0', 'S0']:
        sc = np.array([r['score'] for r in prec[name]])
        all_scores[name] = sc
        st = dist_stats(sc)
        emit(f'{name:<4}{len(sc):>9}'
             + ''.join(f'{st[k]:>9.4f}' for k in keys))
    d_c, d_s = dist_stats(all_scores['C0']), dist_stats(all_scores['S0'])
    emit(f'{"d":<4}{len(all_scores["S0"]) - len(all_scores["C0"]):>9}'
         + ''.join(f'{d_s[k] - d_c[k]:>+9.4f}' for k in keys))

    # ---------------- Part 1.2 TP/FP decomposition -----------------
    emit('\n' + '=' * 76)
    emit('PART 1.2 TP / FP SCORE DECOMPOSITION (StrictMatch@2)')
    emit(f'{"m":<4}{"TP":>8}{"FP":>9}{"TPmean":>9}{"FPmean":>9}'
         f'{"TPmed":>9}{"FPmed":>9}')
    tpfp = {}
    for name in ['C0', 'S0']:
        tp = np.array([r['score'] for r in prec[name] if r['matched']])
        fp = np.array([r['score'] for r in prec[name]
                       if not r['matched']])
        tpfp[name] = (tp, fp)
        emit(f'{name:<4}{len(tp):>8}{len(fp):>9}'
             f'{tp.mean():>9.4f}{fp.mean():>9.4f}'
             f'{np.median(tp):>9.4f}{np.median(fp):>9.4f}')
    emit(f'{"d":<4}{len(tpfp["S0"][0]) - len(tpfp["C0"][0]):>8}'
         f'{len(tpfp["S0"][1]) - len(tpfp["C0"][1]):>9}'
         f'{tpfp["S0"][0].mean() - tpfp["C0"][0].mean():>+9.4f}'
         f'{tpfp["S0"][1].mean() - tpfp["C0"][1].mean():>+9.4f}'
         f'{np.median(tpfp["S0"][0]) - np.median(tpfp["C0"][0]):>+9.4f}'
         f'{np.median(tpfp["S0"][1]) - np.median(tpfp["C0"][1]):>+9.4f}')

    # ---------------- Part 1.3 score-bin table ---------------------
    emit('\n' + '=' * 76)
    emit('PART 1.3 SCORE-BIN TABLE (precision / recall contribution)')
    emit(f'{"bin":<10}{"m":<4}{"count":>9}{"TP":>8}{"FP":>9}'
         f'{"prec":>8}{"recContr":>9}')
    for lo, hi in SCORE_BINS:
        for name in ['C0', 'S0']:
            recs = [r for r in prec[name]
                    if lo <= r['score'] < hi]
            tp = sum(r['matched'] for r in recs)
            cnt = len(recs)
            emit(f'[{lo:.1f},{min(hi, 1.0):.1f}){"":<3}{name:<4}'
                 f'{cnt:>9}{tp:>8}{cnt - tp:>9}'
                 f'{tp / max(cnt, 1):>8.4f}{tp / n:>9.5f}')

    # ---------------- Part 1.4 evidence group calibration ----------
    emit('\n' + '=' * 76)
    emit('PART 1.4 EVIDENCE-GROUP CALIBRATION (TP score by matched-GT '
         'group)')
    emit('note: FP predictions cannot be assigned an evidence group '
         '(predictions carry no per-instance evidence); FP statistics '
         'are therefore reported overall only (Part 1.2).')
    emit(f'{"group":<6}{"m":<4}{"TP":>8}{"TPmean":>9}{"TPmed":>9}'
         f'{"TPp25":>9}{"TPp75":>9}')
    for gname in ['Low', 'Mid', 'High']:
        gsel = group == gname
        for name in ['C0', 'S0']:
            tp = np.array([r['score'] for r in prec[name]
                           if r['matched'] and gsel[r['gt']]])
            emit(f'{gname:<6}{name:<4}{len(tp):>8}'
                 f'{tp.mean():>9.4f}{np.median(tp):>9.4f}'
                 f'{np.percentile(tp, 25):>9.4f}'
                 f'{np.percentile(tp, 75):>9.4f}')

    # ---------------- Part 2.1 zero-point improvement --------------
    emit('\n' + '=' * 76)
    emit('PART 2.1 ZERO-POINT GT DETECTION IMPROVEMENT '
         '(evidence_group=Low, num_lidar_pts==0)')
    zp = (group == 'Low') & (num_pts == 0)
    nzp = int(zp.sum())
    emit(f'GT count: {nzp}')
    emit(f'{"m":<4}{"Strct@.5":>9}{"Strct@1":>9}{"Strct@2":>9}'
         f'{"TP@2":>8}')
    for name in ['C0', 'S0']:
        tp2 = int((zp & strict[(name, 2.0)][0]).sum())
        emit(f'{name:<4}'
             f'{(zp & strict[(name, 0.5)][0]).sum() / nzp:>9.4f}'
             f'{(zp & strict[(name, 1.0)][0]).sum() / nzp:>9.4f}'
             f'{tp2 / nzp:>9.4f}{tp2:>8}')
    new_tp = int((zp & strict[('S0', 2.0)][0]).sum()
                 - (zp & strict[('C0', 2.0)][0]).sum())
    emit(f'new TP (S0 TP - C0 TP): {new_tp:+d}')
    both = zp & strict[('C0', 2.0)][0] & strict[('S0', 2.0)][0]
    only_s0 = zp & ~strict[('C0', 2.0)][0] & strict[('S0', 2.0)][0]
    only_c0 = zp & strict[('C0', 2.0)][0] & ~strict[('S0', 2.0)][0]
    emit(f'matched by both: {int(both.sum())}; S0-only: '
         f'{int(only_s0.sum())}; C0-only: {int(only_c0.sum())}')

    # ---------------- Part 2.2 spatial neighborhood ----------------
    emit('\n' + '=' * 76)
    emit('PART 2.2 SPATIAL NEIGHBORHOOD OF ZERO-POINT LOW GTs')
    emit('for each zero-point Low GT: instances within radius r in the '
         'same frame (GT context identical for C0/S0; predictions '
         'method-specific, any class, score>=0.1)')
    gt_by_tok = defaultdict(list)
    for i, r in enumerate(gt_rows):
        gt_by_tok[r['token']].append(i)
    pred_by_tok = {'C0': defaultdict(list), 'S0': defaultdict(list)}
    for name in ['C0', 'S0']:
        for r in prec[name]:
            pred_by_tok[name][r['token']].append((r['x'], r['y']))

    zp_idx = np.where(zp)[0]
    emit(f'{"r":>5}{"GTany":>16}{"GTnonzero":>16}'
         f'{"predC0":>16}{"predS0":>16}')
    for rad in NB_RADII:
        stats_row = []
        for kind in ['gt_any', 'gt_nz', 'p_c0', 'p_s0']:
            cnts = np.zeros(len(zp_idx))
            for k, i in enumerate(zp_idx):
                r0 = gt_rows[i]
                if kind == 'gt_any':
                    ids = [j for j in gt_by_tok[r0['token']] if j != i]
                    xy = np.array([[gt_rows[j]['x'], gt_rows[j]['y']]
                                   for j in ids]) if ids else \
                        np.zeros((0, 2))
                elif kind == 'gt_nz':
                    ids = [j for j in gt_by_tok[r0['token']]
                           if j != i and gt_rows[j]['num_pts'] > 0]
                    xy = np.array([[gt_rows[j]['x'], gt_rows[j]['y']]
                                   for j in ids]) if ids else \
                        np.zeros((0, 2))
                else:
                    src = pred_by_tok['C0' if kind == 'p_c0'
                                      else 'S0'][r0['token']]
                    xy = np.array(src) if src else np.zeros((0, 2))
                if len(xy):
                    cnts[k] = (np.hypot(xy[:, 0] - r0['x'],
                                        xy[:, 1] - r0['y']) <= rad).sum()
            stats_row.append(cnts)
        emit(f'{rad:>5.1f}'
             + ''.join(f'{c.mean():>7.3f}/{(c > 0).mean():>7.4f}'
                       for c in stats_row))
    emit('  (each cell: mean count / fraction of GTs with >=1 within r; '
         'columns: any other GT | non-zero-point GT | C0 preds | '
         'S0 preds)')

    # ---------------- Part 2.3 frame-level -------------------------
    emit('\n' + '=' * 76)
    emit('PART 2.3 FRAME-LEVEL CONTRIBUTION (zero-point Low GTs)')
    tok2zp = defaultdict(lambda: [0, 0, 0])  # gt, tp_c0, tp_s0
    for i in zp_idx:
        t = gt_rows[i]['token']
        tok2zp[t][0] += 1
        tok2zp[t][1] += int(strict[('C0', 2.0)][0][i])
        tok2zp[t][2] += int(strict[('S0', 2.0)][0][i])
    rows = sorted(tok2zp.items(),
                  key=lambda kv: (-(kv[1][2] - kv[1][1]), kv[0]))
    emit(f'frames with zero-point Low GTs: {len(rows)} '
         f'(of {len(gt_by_tok)} frames with any GT)')
    emit(f'total new TP: {new_tp:+d}; top-20 frames account for '
         f'{sum(v[2] - v[1] for _, v in rows[:20]):+d}')
    emit(f'{"token":<36}{"zpGT":>6}{"TP_C0":>7}{"TP_S0":>7}{"delta":>7}')
    for t, (g, a, b) in rows[:20]:
        emit(f'{t:<36}{g:>6}{a:>7}{b:>7}{b - a:>+7}')
    n_pos = sum(1 for _, v in rows if v[2] - v[1] > 0)
    n_neg = sum(1 for _, v in rows if v[2] - v[1] < 0)
    emit(f'frames with delta>0: {n_pos}; delta<0: {n_neg}; '
         f'delta==0: {len(rows) - n_pos - n_neg}')
    emit('scene-level analysis: Unavailable from current dump '
         '(no scene id stored)')

    # ---------------- Part 2.4 multi-frame visibility --------------
    emit('\n' + '=' * 76)
    emit('PART 2.4 MULTI-FRAME VISIBILITY CHECK')
    emit('Unavailable from current dump: the dumps store only '
         'per-sample-token predictions and GT rows (class, points, box, '
         'lidar_range, heatmap center score). They contain no scene id, '
         'timestamp, sample index or sweep linkage, so whether a '
         'zero-point GT is observed in neighbouring frames cannot be '
         'determined from these files.')

    # ---------------- limitations + summary ------------------------
    emit('\n' + '=' * 76)
    emit('LIMITATIONS')
    emit('  - FP predictions cannot be assigned an evidence group; '
         'per-group FP calibration is therefore not reported.')
    emit('  - scene-level and multi-frame visibility analyses are '
         'unavailable from the current dumps (no scene/temporal '
         'fields).')
    emit('  - zero-point GTs never entered weighted supervision; any '
         'change on them is an indirect effect, and this analysis '
         'only describes co-occurring context, not the information '
         'pathway.')

    emit('\n' + '=' * 76)
    emit('FACTUAL SUMMARY (statistics only; no causal claims)')
    d_pred = len(all_scores['S0']) - len(all_scores['C0'])
    d_tp = len(tpfp['S0'][0]) - len(tpfp['C0'][0])
    d_fp = len(tpfp['S0'][1]) - len(tpfp['C0'][1])
    emit(f'  - Observed prediction count change (score>=0.1): '
         f'{d_pred:+d} (TP {d_tp:+d}, FP {d_fp:+d}); the precision drop '
         f'co-occurs with an FP increase much larger than the TP '
         f'increase.')
    emit(f'  - Statistics show TP mean score '
         f'{tpfp["C0"][0].mean():.4f} -> {tpfp["S0"][0].mean():.4f} and '
         f'FP mean score {tpfp["C0"][1].mean():.4f} -> '
         f'{tpfp["S0"][1].mean():.4f}.')
    emit(f'  - Observed TP mean score by group (C0->S0) is reported in '
         f'Part 1.4; per-group FP calibration is unavailable (see '
         f'limitations).')
    emit(f'  - Observed zero-point Low GTs: {nzp}; StrictMatch@2 '
         f'{(zp & strict[("C0", 2.0)][0]).sum() / nzp:.4f} -> '
         f'{(zp & strict[("S0", 2.0)][0]).sum() / nzp:.4f} '
         f'(new TP {new_tp:+d}; S0-only {int(only_s0.sum())}, '
         f'C0-only {int(only_c0.sum())}).')
    emit('  - Observed spatial context of zero-point Low GTs '
         '(co-occurring GTs and predictions within 1/2/5 m) is '
         'reported in Part 2.2; scene and multi-frame visibility are '
         'unavailable from the current dumps.')

    with open(args.out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nreport saved to {args.out}')


if __name__ == '__main__':
    main()
