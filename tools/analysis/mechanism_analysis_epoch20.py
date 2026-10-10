"""Mechanism analysis for Center-Peak Weighting (epoch-20 C0-full vs S0-full).

Answers three questions using only the existing epoch-20 dumps:
  Q1: is the extra training supervision (peak weight - 1) concentrated on
      low/mid evidence instances?          -> weight mass audit
  Q2: does the performance change come from recall, precision or
      localization?                        -> detection decomposition
  Q3: within Low evidence, which objects are recoverable and which are
      not?                                 -> point-bucket recoverability

Protocol is identical to evidence_stratified_matching_audit.py:
  - same dumps, same GT table, same evidence formula and per-class
    quantile grouping (fixed assignment, shared by C0/S0);
  - score threshold 0.1; greedy one-to-one class-aware StrictMatch;
  - frame-cluster bootstrap, 1000 reps, seed 3407.

Training peak weight (centerpoint_head.py):
    sparse_score = clip(-e_i, 0, sparse_score_max)        # smax = 2.0
    w_i          = clip(1 + lambda * sparse_score, 1, wmax)
    lambda = 0.25, wmax = 1.5
Extra supervision mass: M_g = sum(w_i - 1); share R_g = M_g / sum_i(w_i-1).

Note: NuScenesDataset filters training GTs with num_lidar_pts > 0
(filter_empty_gt), so zero-point GTs receive no training supervision.
Part 1 therefore reports both the full fixed GT table and the
training-supervised subset (num_pts > 0).

This script reads existing files only; it modifies nothing.
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

DIST_BANDS = [('0-20m', 0.0, 20.0), ('20-40m', 20.0, 40.0),
              ('>40m', 40.0, np.inf)]
POINT_BUCKETS = [('Z(0 pts)', lambda p: p == 0),
                 ('P1(1-5)', lambda p: (p >= 1) & (p <= 5)),
                 ('P2(6-20)', lambda p: (p >= 6) & (p <= 20)),
                 ('P3(>20)', lambda p: p > 20)]


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
                         'mechanism_analysis_epoch20.txt')
    return p.parse_args()


def peak_weights(evidence, lam, smax, wmax):
    sparse = np.clip(-evidence, 0.0, smax)
    return np.clip(1.0 + lam * sparse, 1.0, wmax)


def boot_ratio_delta(tok_g, tok_a, tok_b):
    """Frame-cluster bootstrap CI of (rate_b - rate_a); rate = cnt/gt."""
    rng = np.random.default_rng(BOOT_SEED)
    ntok = len(tok_g)
    ds = []
    for _ in range(N_BOOT):
        s = rng.integers(0, ntok, ntok)
        g = tok_g[s].sum()
        if g <= 0:
            continue
        ds.append(tok_b[s].sum() / g - tok_a[s].sum() / g)
    ds = np.asarray(ds)
    return float(ds.mean()), float(np.percentile(ds, 2.5)), \
        float(np.percentile(ds, 97.5))


def boot_prec_delta(tok_tp_a, tok_p_a, tok_tp_b, tok_p_b):
    """Frame-cluster bootstrap CI of precision delta (TP/preds)."""
    rng = np.random.default_rng(BOOT_SEED)
    ntok = len(tok_tp_a)
    ds = []
    for _ in range(N_BOOT):
        s = rng.integers(0, ntok, ntok)
        pa, pb = tok_p_a[s].sum(), tok_p_b[s].sum()
        if pa <= 0 or pb <= 0:
            continue
        ds.append(tok_tp_b[s].sum() / pb - tok_tp_a[s].sum() / pa)
    ds = np.asarray(ds)
    return float(ds.mean()), float(np.percentile(ds, 2.5)), \
        float(np.percentile(ds, 97.5))


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    er = cfg.model.train_cfg.pts.evidence_radius
    stats = dict(
        mu_count=list(er['mu_count']), std_count=list(er['std_count']),
        mu_density=list(er['mu_density']),
        std_density=list(er['std_density']), beta=float(er['beta']))
    lam = float(er.get('peak_weight_lambda', 0.25))
    smax = float(er.get('sparse_score_max', 2.0))
    wmax = float(er.get('peak_weight_max', 1.5))

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
        assert abs(a['x'] - b['x']) < 1e-4 and abs(a['y'] - b['y']) < 1e-4
    n = len(gt_rows)

    cls_ids = np.array([r['cls_id'] for r in gt_rows])
    tokens = np.array([r['token'] for r in gt_rows])
    num_pts = np.array([r['num_pts'] for r in gt_rows])
    lidar_range = np.array([r['lidar_range'] for r in gt_rows])
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

    weights = peak_weights(evidence, lam, smax, wmax)

    strict = {}
    for name, dump in [('C0', c0), ('S0', s0)]:
        for tau in TAUS:
            strict[(name, tau)] = strict_match(dump, gt_rows, tau)
    m2 = {'C0': strict[('C0', 2.0)][0], 'S0': strict[('S0', 2.0)][0]}
    d2 = {'C0': strict[('C0', 2.0)][1], 'S0': strict[('S0', 2.0)][1]}

    # per-token prediction counts (score >= 0.1) and prediction coords
    # per (token, class) for failure decomposition
    pred_cnt = {'C0': {}, 'S0': {}}
    preds_tc = {'C0': defaultdict(dict), 'S0': defaultdict(dict)}
    for name, dump in [('C0', c0), ('S0', s0)]:
        for token, pr in dump['preds'].items():
            keep = pr['scores'] >= SCORE_THR
            pred_cnt[name][token] = int(keep.sum())
            for lbl in np.unique(pr['labels'][keep]):
                lsel = keep & (pr['labels'] == lbl)
                preds_tc[name][token][int(lbl)] = (
                    pr['boxes'][lsel][:, :2], pr['scores'][lsel])

    # frames with zero GT rows still contribute predictions to the
    # precision denominator (same protocol as the main analysis, whose
    # n_pred_used counts all dump tokens); include them in the token set
    pred_tokens = np.array(
        list(c0['preds'].keys()) + list(s0['preds'].keys()),
        dtype=tokens.dtype)
    uniq_tokens = np.unique(np.concatenate([tokens, pred_tokens]))
    ntok = len(uniq_tokens)
    tok_idx = np.searchsorted(uniq_tokens, tokens)
    tok_pred = {name: np.array([pred_cnt[name].get(t, 0)
                                for t in uniq_tokens], dtype=float)
                for name in ['C0', 'S0']}

    def token_counts(mask, *matched):
        out = [np.bincount(tok_idx[mask], minlength=ntok).astype(float)]
        for m in matched:
            out.append(np.bincount(tok_idx[mask & m],
                                   minlength=ntok).astype(float))
        return out

    lines = []

    def emit(s=''):
        print(s)
        lines.append(s)

    emit('MECHANISM ANALYSIS (epoch-20): C0-full (baseline) vs '
         'S0-full (peak_weight lambda=0.25)')
    emit('=' * 76)
    emit(f'input C0 dump:      {args.c0}')
    emit(f'input S0 dump:      {args.s0}')
    emit(f'analysis script:    {Path(__file__).resolve()}')
    emit(f'config:             {args.config}')
    emit(f'GT count: {n} (tables identical between dumps)')
    emit(f'group sizes: Low={int((group == "Low").sum())} '
         f'Mid={int((group == "Mid").sum())} '
         f'High={int((group == "High").sum())}')
    emit('')
    emit('FORMULAS')
    emit('  evidence e_i = beta*z_count + (1-beta)*z_density on '
         'log1p(count), log1p(density); per-class mu/std from config; '
         f'beta={stats["beta"]}')
    emit('  peak weight w_i = clip(1 + lambda*clip(-e_i,0,smax), 1, wmax),'
         f' lambda={lam}, smax={smax}, wmax={wmax}')
    emit('  extra supervision mass M_g = sum_{i in g}(w_i - 1); '
         'share R_g = M_g / sum_i(w_i - 1)')
    emit('  StrictMatch@tau: per frame per class, preds score>='
         f'{SCORE_THR} sorted by score desc, greedy one-to-one nearest '
         'unmatched same-class GT with BEV d<=tau')
    emit('  Recall_g = #(GT in g matched under StrictMatch@2) / #GT(g) '
         '(identical to StrictMatch@2 of the audit)')
    emit('  Precision = TP / #predictions(score>=0.1); per-group '
         'Precision_g = TP_g / #predictions, i.e. group contribution to '
         'overall precision (shares the global denominator, sums over g '
         'to overall precision); predictions carry no evidence label')
    emit('  SCMR_cond@0.5 = #(matched@2 with d<=0.5) / #matched@2')
    emit('  mCE@2m = mean BEV center distance over StrictMatch@2 pairs')
    emit(f'  bootstrap: frame-cluster (resample sample tokens), '
         f'{N_BOOT} reps, seed={BOOT_SEED}, 95% percentile CI of '
         'delta (S0-C0)')
    emit('  note: NuScenesDataset filter_empty_gt filters training GTs '
         'with num_lidar_pts > 0; zero-point GTs receive no training '
         'supervision (their w_i is never applied)')

    # ---------------- Part 1: weight mass audit ----------------
    total_mass = float((weights - 1.0).sum())

    def wstats(name, sel, ref_mass):
        cnt = int(sel.sum())
        if cnt == 0:
            return None
        w = weights[sel]
        e = evidence[sel]
        mass = float((w - 1.0).sum())
        flag = ' *low*' if cnt < LOW_SUPPORT else ''
        emit(f'{name:<22}{cnt:>8}{cnt / n:>9.4f}'
             f'{e.mean():>9.3f}{np.median(e):>9.3f}'
             f'{w.mean():>8.4f}{np.median(w):>8.4f}{w.max():>8.4f}'
             f'{(w == 1.0).mean():>8.4f}'
             f'{((w > 1.0) & (w < wmax)).mean():>8.4f}'
             f'{(w >= wmax).mean():>8.4f}'
             f'{mass:>11.1f}{mass / max(ref_mass, 1e-12):>8.4f}{flag}')
        return mass

    def weight_table(title, items, ref_mass):
        emit(f'\n=== {title} ===')
        emit(f'{"group":<22}{"n":>8}{"pct":>9}'
             f'{"evMean":>9}{"evMed":>9}'
             f'{"wMean":>8}{"wMed":>8}{"wMax":>8}'
             f'{"w==1":>8}{"1<w<cap":>8}{"w==cap":>8}'
             f'{"M_g":>11}{"R_g":>8}')
        for name, sel in items:
            wstats(name, sel, ref_mass)

    emit('\n' + '=' * 76)
    emit('PART 1. WEIGHT MASS AUDIT (w_i computed on the fixed val GT '
         'table with training statistics)')

    groups_sel = [(g, group == g) for g in ['Low', 'Mid', 'High']]
    weight_table('1a. BY EVIDENCE GROUP (all GT)', groups_sel, total_mass)

    sup = num_pts > 0
    sup_mass = float((weights[sup] - 1.0).sum())
    emit(f'\ntraining-supervised subset (num_lidar_pts>0): '
         f'n={int(sup.sum())}, mass={sup_mass:.1f} '
         f'({sup_mass / total_mass:.4f} of total); zero-point GTs hold '
         f'{1.0 - sup_mass / total_mass:.4f} of total (never applied '
         'in training)')
    weight_table(
        '1b. BY EVIDENCE GROUP (supervised subset, num_pts>0)',
        [(g, (group == g) & sup) for g in ['Low', 'Mid', 'High']],
        sup_mass)

    weight_table(
        '1c. BY DISTANCE (all GT, R_g relative to total mass)',
        [(nm, (lidar_range >= lo) & (lidar_range < hi))
         for nm, lo, hi in DIST_BANDS], total_mass)

    weight_table(
        '1d. BY CLASS (all GT, R_g relative to total mass)',
        [(cn, cls_ids == class_names.index(cn)) for cn in FOCUS_CLASSES],
        total_mass)

    # ---------------- Part 2: recall/precision/localization ----------
    emit('\n' + '=' * 76)
    emit('PART 2. RECALL / PRECISION / LOCALIZATION DECOMPOSITION '
         '(StrictMatch@2, score>=0.1)')
    n_pred = {name: float(tok_pred[name].sum()) for name in ['C0', 'S0']}

    emit(f'\npredictions(score>=0.1): C0={int(n_pred["C0"])} '
         f'S0={int(n_pred["S0"])}')
    emit(f'{"group":<8}{"m":<4}{"n_GT":>8}{"TP":>8}{"Recall":>9}'
         f'{"Prec":>9}{"Strict@.5":>10}{"Strict@1":>10}'
         f'{"dMean":>8}{"dMed":>8}{"dP75":>8}{"dP90":>8}')
    part2_sel = [('Overall', np.ones(n, dtype=bool))] + \
        [(g, group == g) for g in ['Low', 'Mid', 'High']]
    for gname, sel in part2_sel:
        for mname in ['C0', 'S0']:
            mm = sel & m2[mname]
            md = d2[mname][mm]
            emit(f'{gname:<8}{mname:<4}{int(sel.sum()):>8}'
                 f'{int(mm.sum()):>8}'
                 f'{mm.sum() / sel.sum():>9.4f}'
                 f'{mm.sum() / n_pred[mname]:>9.4f}'
                 f'{(sel & strict[(mname, 0.5)][0]).sum() / sel.sum():>10.4f}'
                 f'{(sel & strict[(mname, 1.0)][0]).sum() / sel.sum():>10.4f}'
                 f'{md.mean() if len(md) else float("nan"):>8.4f}'
                 f'{np.median(md) if len(md) else float("nan"):>8.4f}'
                 f'{np.percentile(md, 75) if len(md) else float("nan"):>8.4f}'
                 f'{np.percentile(md, 90) if len(md) else float("nan"):>8.4f}')

    emit(f'\nbootstrap 95% CI of delta (S0-C0), frame-cluster, '
         f'{N_BOOT} reps, seed={BOOT_SEED}')
    emit(f'{"group":<8}{"dRecall":>28}{"dPrecision":>28}')
    for gname, sel in part2_sel:
        tg, ta, tb = token_counts(sel, m2['C0'], m2['S0'])
        rc = boot_ratio_delta(tg, ta, tb)
        pc = boot_prec_delta(ta, tok_pred['C0'], tb, tok_pred['S0'])
        emit(f'{gname:<8}'
             f'  {rc[0]:+.4f} [{rc[1]:+.4f},{rc[2]:+.4f}]'
             f'  {pc[0]:+.4f} [{pc[1]:+.4f},{pc[2]:+.4f}]')

    # ---------------- Part 3: Low recoverability ----------------
    emit('\n' + '=' * 76)
    emit('PART 3. LOW-EVIDENCE RECOVERABILITY BY POINT COUNT '
         '(evidence_group == Low)')
    low = group == 'Low'
    low_mass = float((weights[low] - 1.0).sum())

    emit(f'\n{"bucket":<10}{"n":>8}{"distMean":>9}{"distMed":>8}'
         f'{"evMean":>8}{"evMed":>8}{"wMean":>8}{"wMax":>7}'
         f'{"M_g":>9}{"R_g|Low":>9}')
    bucket_sels = []
    for bname, fn in POINT_BUCKETS:
        sel = low & fn(num_pts)
        bucket_sels.append((bname, sel))
        cnt = int(sel.sum())
        flag = ' *low*' if cnt < LOW_SUPPORT else ''
        w = weights[sel]
        emit(f'{bname:<10}{cnt:>8}'
             f'{lidar_range[sel].mean():>9.1f}'
             f'{np.median(lidar_range[sel]):>8.1f}'
             f'{evidence[sel].mean():>8.3f}'
             f'{np.median(evidence[sel]):>8.3f}'
             f'{w.mean() if cnt else float("nan"):>8.4f}'
             f'{w.max() if cnt else float("nan"):>7.4f}'
             f'{(w - 1.0).sum():>9.1f}'
             f'{(w - 1.0).sum() / max(low_mass, 1e-12):>9.4f}{flag}')

    emit(f'\n{"bucket":<10}{"m":<4}{"n":>7}{"Strct@.5":>9}{"Strct@1":>9}'
         f'{"Strct@2":>9}{"SCMRc@.5":>10}{"mCE@2m":>8}')
    for bname, sel in bucket_sels:
        for mname in ['C0', 'S0']:
            cnt = int(sel.sum())
            if cnt == 0:
                continue
            mm = sel & m2[mname]
            md = d2[mname][mm]
            flag = ' *low*' if cnt < LOW_SUPPORT else ''
            emit(f'{bname:<10}{mname:<4}{cnt:>7}'
                 f'{(sel & strict[(mname, 0.5)][0]).sum() / cnt:>9.4f}'
                 f'{(sel & strict[(mname, 1.0)][0]).sum() / cnt:>9.4f}'
                 f'{mm.sum() / cnt:>9.4f}'
                 f'{(mm & (d2[mname] <= 0.5)).sum() / max(mm.sum(), 1):>10.4f}'
                 f'{md.mean() if len(md) else float("nan"):>8.4f}{flag}')

    emit(f'\nbootstrap 95% CI of delta StrictMatch@2 (S0-C0), '
         f'frame-cluster, {N_BOOT} reps, seed={BOOT_SEED}')
    for bname, sel in bucket_sels:
        if sel.sum() < LOW_SUPPORT:
            emit(f'{bname:<10} low-support (n={int(sel.sum())}), '
                 'CI not reported')
            continue
        tg, ta, tb = token_counts(sel, m2['C0'], m2['S0'])
        rc = boot_ratio_delta(tg, ta, tb)
        emit(f'{bname:<10}  {rc[0]:+.4f} [{rc[1]:+.4f},{rc[2]:+.4f}]')

    # ---------------- Part 4: Low distance controlled --------------
    emit('\n' + '=' * 76)
    emit('PART 4. LOW x DISTANCE x POINT BUCKET (StrictMatch@2)')
    emit(f'{"dist":<8}{"bucket":<10}{"n":>7}{"C0":>8}{"S0":>8}'
         f'{"delta":>8}')
    for dname, lo, hi in DIST_BANDS:
        dsel = low & (lidar_range >= lo) & (lidar_range < hi)
        for bname, fn in POINT_BUCKETS:
            sel = dsel & fn(num_pts)
            cnt = int(sel.sum())
            if cnt == 0:
                continue
            r_c0 = (sel & m2['C0']).sum() / cnt
            r_s0 = (sel & m2['S0']).sum() / cnt
            flag = ' *low*' if cnt < LOW_SUPPORT else ''
            emit(f'{dname:<8}{bname:<10}{cnt:>7}{r_c0:>8.4f}'
                 f'{r_s0:>8.4f}{r_s0 - r_c0:>+8.4f}{flag}')

    # ---------------- Part 5: Low bicycle --------------------------
    emit('\n' + '=' * 76)
    emit('PART 5. LOW-EVIDENCE BICYCLE TARGETED ANALYSIS')
    bic = low & (cls_ids == class_names.index('bicycle'))
    nb = int(bic.sum())
    emit(f'\nGT count: {nb}')
    emit('point distribution: ' + ', '.join(
        f'{bname}={int((bic & fn(num_pts)).sum())}'
        for bname, fn in POINT_BUCKETS))
    emit('distance distribution: ' + ', '.join(
        f'{dname}={int((bic & (lidar_range >= lo) & (lidar_range < hi)).sum())}'
        for dname, lo, hi in DIST_BANDS))
    wb = weights[bic]
    emit(f'weight: mean={wb.mean():.4f} median={np.median(wb):.4f} '
         f'max={wb.max():.4f} frac(w>1)={(wb > 1.0).mean():.4f}')

    emit(f'\n{"m":<4}{"Strct@.5":>9}{"Strct@1":>9}{"Strct@2":>9}'
         f'{"dMean":>8}{"dMed":>8}{"dP75":>8}{"dP90":>8}')
    for mname in ['C0', 'S0']:
        mm = bic & m2[mname]
        md = d2[mname][mm]
        emit(f'{mname:<4}'
             f'{(bic & strict[(mname, 0.5)][0]).sum() / nb:>9.4f}'
             f'{(bic & strict[(mname, 1.0)][0]).sum() / nb:>9.4f}'
             f'{mm.sum() / nb:>9.4f}'
             f'{md.mean():>8.4f}{np.median(md):>8.4f}'
             f'{np.percentile(md, 75):>8.4f}{np.percentile(md, 90):>8.4f}')

    emit(f'\nbootstrap 95% CI of delta (S0-C0), frame-cluster, '
         f'{N_BOOT} reps, seed={BOOT_SEED}')
    for tau in TAUS:
        tg, ta, tb = token_counts(bic, strict[('C0', tau)][0],
                                  strict[('S0', tau)][0])
        rc = boot_ratio_delta(tg, ta, tb)
        emit(f'StrictM@{tau:<4}  {rc[0]:+.4f} [{rc[1]:+.4f},{rc[2]:+.4f}]')

    # failure decomposition
    emit('\nfailure decomposition (per method, same GT set):')
    emit('  cat1: no bicycle prediction (score>=0.1) within 2m of the GT')
    emit('  cat2a: bicycle prediction within 2m exists, GT matched@2 '
         'but d>0.5')
    emit('  cat2b: bicycle prediction within 2m exists, but GT not '
         'matched under StrictMatch@2 (preds consumed by other GTs)')
    emit('  cat3: non-bicycle competition - other-class predictions '
         'within 2m of the GT')
    bic_idx = np.where(bic)[0]
    other_cls = {'pedestrian': class_names.index('pedestrian'),
                 'motorcycle': class_names.index('motorcycle'),
                 'traffic_cone': class_names.index('traffic_cone')}
    emit(f'{"m":<4}{"cat1":>12}{"cat2a":>12}{"cat2b":>12}'
         f'{"cat3_any":>12}{"ped":>7}{"moto":>7}{"cone":>7}{"other":>7}')
    for mname in ['C0', 'S0']:
        c1 = c2a = c2b = c3 = 0
        comp = defaultdict(int)
        for i in bic_idx:
            r = gt_rows[i]
            tc = preds_tc[mname].get(r['token'], {})
            gxy = np.array([r['x'], r['y']])
            has_bic = False
            if class_names.index('bicycle') in tc:
                xy, _ = tc[class_names.index('bicycle')]
                has_bic = bool((np.hypot(xy[:, 0] - gxy[0],
                                         xy[:, 1] - gxy[1]) <= 2.0).any())
            if not has_bic:
                c1 += 1
            elif m2[mname][i] and d2[mname][i] > 0.5:
                c2a += 1
            elif not m2[mname][i]:
                c2b += 1
            any_other = False
            for lbl, (xy, _) in tc.items():
                if lbl == class_names.index('bicycle'):
                    continue
                k = int((np.hypot(xy[:, 0] - gxy[0],
                                  xy[:, 1] - gxy[1]) <= 2.0).sum())
                if k:
                    any_other = True
                    comp[lbl] += k
            c3 += int(any_other)
        ped = comp.get(other_cls['pedestrian'], 0)
        mot = comp.get(other_cls['motorcycle'], 0)
        cone = comp.get(other_cls['traffic_cone'], 0)
        oth = sum(v for k, v in comp.items()
                  if k not in other_cls.values())
        emit(f'{mname:<4}{c1:>7}/{nb:<4}{c2a:>7}/{nb:<4}{c2b:>7}/{nb:<4}'
             f'{c3:>7}/{nb:<4}{ped:>7}{mot:>7}{cone:>7}{oth:>7}')
    emit('  (cat3 ped/moto/cone/other = total nearby other-class '
         'prediction count summed over Low bicycle GTs)')

    # ---------------- bootstrap config + summary -------------------
    emit('\n' + '=' * 76)
    emit('BOOTSTRAP CONFIGURATION')
    emit(f'  unit: frame cluster (sample tokens, pool their GTs/preds); '
         f'reps={N_BOOT}; seed={BOOT_SEED}; 95% percentile CI')
    emit('  C0 and S0 share the same resample indices (paired)')

    emit('\n' + '=' * 76)
    emit('FACTUAL SUMMARY (statistics only; no causal claims)')
    gs = {}
    for gname in ['Low', 'Mid', 'High']:
        sel = group == gname
        gs[gname] = dict(
            mass=float((weights[sel] - 1.0).sum()),
            rec_c0=(sel & m2['C0']).sum() / sel.sum(),
            rec_s0=(sel & m2['S0']).sum() / sel.sum())
    emit(f'  - Observed weight mass share R_g: '
         f'Low={gs["Low"]["mass"] / total_mass:.4f}, '
         f'Mid={gs["Mid"]["mass"] / total_mass:.4f}, '
         f'High={gs["High"]["mass"] / total_mass:.4f} '
         f'(full GT table); on the supervised subset (num_pts>0) the '
         f'shares are reported in table 1b.')
    emit(f'  - Results show Recall(StrictMatch@2) deltas: '
         f'Low={gs["Low"]["rec_s0"] - gs["Low"]["rec_c0"]:+.4f}, '
         f'Mid={gs["Mid"]["rec_s0"] - gs["Mid"]["rec_c0"]:+.4f}, '
         f'High={gs["High"]["rec_s0"] - gs["High"]["rec_c0"]:+.4f}; '
         'corresponding bootstrap CIs are reported in Part 2.')
    emit('  - Statistics indicate the localization statistics '
         '(mean/median/p75/p90 center distance, StrictMatch@0.5) per '
         'group are reported in Part 2; the direction and significance '
         'of changes are stated by the CIs, not asserted here.')
    emit('  - Observed recoverability split inside Low: point-bucket '
         'metrics and deltas are reported in Parts 3-4; buckets with '
         f'n<{LOW_SUPPORT} are flagged low-support and excluded from '
         'strong statements.')
    emit('  - Low bicycle detection deltas and failure decomposition '
         'are reported in Part 5.')

    with open(args.out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nreport saved to {args.out}')


if __name__ == '__main__':
    main()
