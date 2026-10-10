"""Evidence-stratified offline diagnosis: C0 (baseline) vs S0 (peak_weight).

Reuses the exact evidence definition of CenterHead.compute_evidence():
    log_count   = log1p(num_lidar_pts)
    log_density = log1p(num_lidar_pts / box_volume)
    z_count     = (log_count   - mu_count[cls])   / std_count[cls]
    z_density   = (log_density - mu_density[cls]) / std_density[cls]
    e           = beta * z_count + (1 - beta) * z_density
with mu/std/beta loaded from the controlled config file itself.

Matching (per sample, per class):
    predictions filtered at score >= 0.1, sorted by score descending;
    greedy one-to-one assignment: each prediction matches the nearest
    unmatched same-class GT within 2.0 m BEV center distance; each GT is
    matched at most once.

Metrics per group g (GT-conditioned):
    Match@tau(g)      = #(GT matched with d <= tau) / #GT(g)
    SCMR_proj@tau(g)  = project historical definition from
                        instance_sparsity_analysis.py:
                        #(matched@2m AND d <= tau) / #GT(g)
                        (identical to Match@tau under this matching)
    SCMR_cond@tau(g)  = #(GT with d <= tau) / #(GT matched@2m in g)
    mCE@2m(g)         = mean BEV center distance over GT matched@2m
    MissRate(g)       = 1 - Match@2m(g)

Grouping: per-class evidence quantiles computed once from validation GT
(Low = bottom 25%, Mid = middle 50%, High = top 25%); the identical
assignment is used for both methods.

Bootstrap: paired GT-instance resampling (same resample for C0 and S0),
1000 repetitions, seed 3407, 95% percentile CI of the S0 - C0 delta.
"""

import argparse

import numpy as np
import pickle
from mmengine.config import Config

SCORE_THR = 0.1
TAUS = [0.5, 1.0, 2.0]
MATCH_DIST = 2.0
N_BOOT = 1000
BOOT_SEED = 3407
LOW_SUPPORT = 30
FOCUS_CLASSES = [
    'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
    'pedestrian', 'bicycle',
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='configs/centerpoint/mycenterpoint_nus_evidence_controlled.py')
    parser.add_argument('--c0', required=True)
    parser.add_argument('--s0', required=True)
    parser.add_argument('--out-prefix', required=True)
    return parser.parse_args()


def compute_evidence(n_pts, volume, cls_id, stats):
    log_count = np.log1p(n_pts)
    log_density = np.log1p(n_pts / volume)
    z_count = ((log_count - stats['mu_count'][cls_id])
               / max(stats['std_count'][cls_id], 1e-6))
    z_density = ((log_density - stats['mu_density'][cls_id])
                 / max(stats['std_density'][cls_id], 1e-6))
    return stats['beta'] * z_count + (1.0 - stats['beta']) * z_density


def build_gt_table(dump):
    rows = []
    for token, gts in dump['gts'].items():
        for gi, g in enumerate(gts):
            rows.append(dict(
                token=token, gt_idx=gi, cls_id=g['cls_id'],
                num_pts=g['num_lidar_pts'], volume=g['volume'],
                x=g['x'], y=g['y'], lidar_range=g['lidar_range'],
                hm_center_score=g['hm_center_score']))
    return rows


def match_method(dump, gt_rows):
    """Greedy one-to-one class-aware matching. Returns per-GT arrays."""
    n = len(gt_rows)
    matched = np.zeros(n, dtype=bool)
    dist = np.full(n, np.nan)

    # index GTs per (token, class)
    from collections import defaultdict
    by_sample_cls = defaultdict(list)
    for i, r in enumerate(gt_rows):
        by_sample_cls[(r['token'], r['cls_id'])].append(i)

    preds_by_sample_cls = defaultdict(list)
    for token, pr in dump['preds'].items():
        keep = pr['scores'] >= SCORE_THR
        for box, score, label in zip(pr['boxes'][keep],
                                     pr['scores'][keep],
                                     pr['labels'][keep]):
            preds_by_sample_cls[(token, int(label))].append(
                (float(score), float(box[0]), float(box[1])))

    n_pred_used = sum(len(v) for v in preds_by_sample_cls.values())

    for key, gt_ids in by_sample_cls.items():
        preds = preds_by_sample_cls.get(key, [])
        if not preds:
            continue
        preds.sort(key=lambda p: -p[0])
        gt_xy = np.array([[gt_rows[i]['x'], gt_rows[i]['y']]
                          for i in gt_ids])
        taken = np.zeros(len(gt_ids), dtype=bool)
        for score, px, py in preds:
            d = np.hypot(gt_xy[:, 0] - px, gt_xy[:, 1] - py)
            d[taken] = np.inf
            j = int(np.argmin(d))
            if d[j] <= MATCH_DIST:
                taken[j] = True
                matched[gt_ids[j]] = True
                dist[gt_ids[j]] = d[j]
    return matched, dist, n_pred_used


def group_metrics(sel, matched, dist):
    """Metrics over a boolean GT selection."""
    n = int(sel.sum())
    if n == 0:
        return None
    m = matched & sel
    out = {'n': n}
    for tau in TAUS:
        hit = sel & matched & (dist <= tau)
        out[f'match@{tau}'] = hit.sum() / n
        denom = max(int(m.sum()), 1)
        out[f'scmr_cond@{tau}'] = hit.sum() / denom
    out['scmr_proj@0.5'] = out['match@0.5']  # project def == Match@tau
    md = dist[m]
    out['n_matched'] = int(m.sum())
    out['mce@2m'] = float(np.mean(md)) if len(md) else float('nan')
    out['medce@2m'] = float(np.median(md)) if len(md) else float('nan')
    out['miss@2m'] = 1.0 - out['match@2.0']
    return out


def fmt(v, pct=True, nd=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '  nan '
    return f'{v:.{nd}f}'


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    er = cfg.model.train_cfg.pts.evidence_radius
    stats = dict(
        mu_count=list(er['mu_count']), std_count=list(er['std_count']),
        mu_density=list(er['mu_density']),
        std_density=list(er['std_density']), beta=float(er['beta']))
    print(f"evidence stats loaded from config: beta={stats['beta']}")

    with open(args.c0, 'rb') as f:
        c0 = pickle.load(f)
    with open(args.s0, 'rb') as f:
        s0 = pickle.load(f)
    class_names = c0['class_names']

    gt_c0 = build_gt_table(c0)
    gt_s0 = build_gt_table(s0)
    assert len(gt_c0) == len(gt_s0)
    for a, b in zip(gt_c0, gt_s0):
        assert a['token'] == b['token'] and a['gt_idx'] == b['gt_idx']
        assert a['cls_id'] == b['cls_id'] and a['num_pts'] == b['num_pts']
        assert abs(a['x'] - b['x']) < 1e-4 and abs(a['y'] - b['y']) < 1e-4
    print(f'GT tables identical between dumps: {len(gt_c0)} GTs')

    n = len(gt_c0)
    cls_ids = np.array([r['cls_id'] for r in gt_c0])
    num_pts = np.array([r['num_pts'] for r in gt_c0], dtype=np.float64)
    volume = np.array([r['volume'] for r in gt_c0])
    lidar_range = np.array([r['lidar_range'] for r in gt_c0])
    evidence = np.array([
        compute_evidence(r['num_pts'], r['volume'], r['cls_id'], stats)
        for r in gt_c0])
    hm_c0 = np.array([r['hm_center_score'] if r['hm_center_score']
                      is not None else np.nan for r in gt_c0])
    hm_s0 = np.array([r['hm_center_score'] if r['hm_center_score']
                      is not None else np.nan for r in gt_s0])

    # per-class evidence quantiles -> group assignment (shared C0/S0)
    group = np.full(n, '', dtype=object)
    thresholds = {}
    for cid, cname in enumerate(class_names):
        sel = cls_ids == cid
        if sel.sum() == 0:
            continue
        q25, q75 = np.quantile(evidence[sel], [0.25, 0.75])
        thresholds[cname] = (float(q25), float(q75))
        group[sel & (evidence <= q25)] = 'Low'
        group[sel & (evidence > q25) & (evidence < q75)] = 'Mid'
        group[sel & (evidence >= q75)] = 'High'

    print('\n=== GT overview ===')
    print(f'total GT: {n}, zero-point GT: {(num_pts == 0).sum()}')
    for cid, cname in enumerate(class_names):
        print(f'  {cname}: {(cls_ids == cid).sum()}')
    for g in ['Low', 'Mid', 'High']:
        sel = group == g
        print(f'  {g}: n={sel.sum()}, '
              f'pts mean/med={num_pts[sel].mean():.1f}/'
              f'{np.median(num_pts[sel]):.0f}, '
              f'evidence mean/med={evidence[sel].mean():.3f}/'
              f'{np.median(evidence[sel]):.3f}, '
              f'range mean/med={lidar_range[sel].mean():.1f}/'
              f'{np.median(lidar_range[sel]):.1f}')

    # matching
    m_c0, d_c0, n_pred_c0 = match_method(c0, gt_c0)
    m_s0, d_s0, n_pred_s0 = match_method(s0, gt_c0)
    print(f'\npreds used (score>={SCORE_THR}): C0={n_pred_c0}, '
          f'S0={n_pred_s0}')
    print(f'unmatched GT rate: C0={1 - m_c0.mean():.4f}, '
          f'S0={1 - m_s0.mean():.4f}')

    lines = []

    def emit(s=''):
        print(s)
        lines.append(s)

    def metric_table(title, selectors):
        emit(f'\n=== {title} ===')
        emit(f'{"group":<12}{"method":<6}{"n":>7}{"M@0.5":>8}{"M@1":>8}'
             f'{"M@2":>8}{"SCMRc@0.5":>10}{"SCMRc@1":>9}{"mCE@2m":>9}'
             f'{"medCE":>8}{"miss@2m":>9}')
        for name, sel in selectors:
            for mname, m, d in [('C0', m_c0, d_c0), ('S0', m_s0, d_s0)]:
                r = group_metrics(sel, m, d)
                if r is None:
                    continue
                flag = ' *low-support*' if r['n'] < LOW_SUPPORT else ''
                emit(f'{name:<12}{mname:<6}{r["n"]:>7}'
                     f'{fmt(r["match@0.5"]):>8}{fmt(r["match@1.0"]):>8}'
                     f'{fmt(r["match@2.0"]):>8}'
                     f'{fmt(r["scmr_cond@0.5"]):>10}'
                     f'{fmt(r["scmr_cond@1.0"]):>9}'
                     f'{fmt(r["mce@2m"]):>9}{fmt(r["medce@2m"]):>8}'
                     f'{fmt(r["miss@2m"]):>9}{flag}')

    all_sel = np.ones(n, dtype=bool)
    metric_table('ALL-CLASS MICRO (evidence groups)',
                 [(g, group == g) for g in ['Low', 'Mid', 'High']]
                 + [('All', all_sel)])

    # macro: mean over classes with support >= LOW_SUPPORT
    emit('\n=== ALL-CLASS MACRO (mean over classes with n>='
         f'{LOW_SUPPORT}) ===')
    for g in ['Low', 'Mid', 'High']:
        for mname, m, d in [('C0', m_c0, d_c0), ('S0', m_s0, d_s0)]:
            vals = {k: [] for k in ['match@0.5', 'match@1.0', 'match@2.0',
                                    'scmr_cond@0.5', 'mce@2m']}
            for cid in range(len(class_names)):
                sel = (group == g) & (cls_ids == cid)
                r = group_metrics(sel, m, d)
                if r is None or r['n'] < LOW_SUPPORT:
                    continue
                for k in vals:
                    vals[k].append(r[k])
            emit(f'{g:<6}{mname:<6}'
                 + '  '.join(f'{k}={np.mean(v):.4f}'
                             for k, v in vals.items()))

    # Low group by distance band
    bands = [('0-20m', (lidar_range >= 0) & (lidar_range < 20)),
             ('20-40m', (lidar_range >= 20) & (lidar_range < 40)),
             ('>40m', lidar_range >= 40)]
    metric_table('LOW EVIDENCE BY DISTANCE',
                 [('Low all', group == 'Low')]
                 + [(f'Low {bn}', (group == 'Low') & bs)
                    for bn, bs in bands])

    # focus classes Low group
    metric_table(
        'FOCUS CLASSES - LOW EVIDENCE',
        [(cname, (group == 'Low')
          & (cls_ids == class_names.index(cname)))
         for cname in FOCUS_CLASSES])

    # auxiliary fixed point buckets
    pb = [('0 pts', num_pts == 0), ('1-5 pts', (num_pts >= 1)
           & (num_pts <= 5)), ('6-20 pts', (num_pts >= 6)
           & (num_pts <= 20)), ('>20 pts', num_pts > 20)]
    metric_table('AUXILIARY FIXED POINT BUCKETS (not main grouping)', pb)

    # heatmap center score diagnostic
    emit('\n=== HEATMAP CENTER SCORE AT GT CENTER CELL (diagnostic) ===')
    for g in ['Low', 'Mid', 'High']:
        sel = (group == g) & ~np.isnan(hm_c0)
        emit(f'{g:<6} n={sel.sum():>6}  '
             f'C0 mean/med={np.nanmean(hm_c0[sel]):.4f}/'
             f'{np.nanmedian(hm_c0[sel]):.4f}  '
             f'S0 mean/med={np.nanmean(hm_s0[sel]):.4f}/'
             f'{np.nanmedian(hm_s0[sel]):.4f}')

    # delta summary
    emit('\n=== DELTA SUMMARY (S0 - C0) ===')
    emit(f'{"group":<12}{"dM@0.5":>9}{"dM@1":>9}{"dM@2":>9}'
         f'{"dSCMRc@0.5":>11}{"dmCE@2m":>9}')
    delta_selectors = [(g, group == g) for g in ['Low', 'Mid', 'High']]
    delta_selectors.append(('All', all_sel))
    for bn, bs in bands:
        delta_selectors.append((f'Low {bn}', (group == 'Low') & bs))
    for name, sel in delta_selectors:
        r0 = group_metrics(sel, m_c0, d_c0)
        r1 = group_metrics(sel, m_s0, d_s0)
        if r0 is None or r1 is None:
            continue
        emit(f'{name:<12}{r1["match@0.5"]-r0["match@0.5"]:>+9.4f}'
             f'{r1["match@1.0"]-r0["match@1.0"]:>+9.4f}'
             f'{r1["match@2.0"]-r0["match@2.0"]:>+9.4f}'
             f'{r1["scmr_cond@0.5"]-r0["scmr_cond@0.5"]:>+11.4f}'
             f'{r1["mce@2m"]-r0["mce@2m"]:>+9.4f}')

    # bootstrap paired CIs for deltas
    emit(f'\n=== BOOTSTRAP 95% CI of delta (S0-C0), {N_BOOT} paired '
         f'resamples, seed={BOOT_SEED} ===')
    rng = np.random.default_rng(BOOT_SEED)

    def boot_delta(sel, stat):
        idx = np.where(sel)[0]
        if len(idx) < LOW_SUPPORT:
            return None
        deltas = []
        for _ in range(N_BOOT):
            b = idx[rng.integers(0, len(idx), len(idx))]
            if stat == 'match@0.5':
                s1 = (m_s0[b] & (d_s0[b] <= 0.5)).mean()
                s2 = (m_c0[b] & (d_c0[b] <= 0.5)).mean()
            elif stat == 'match@1.0':
                s1 = (m_s0[b] & (d_s0[b] <= 1.0)).mean()
                s2 = (m_c0[b] & (d_c0[b] <= 1.0)).mean()
            elif stat == 'match@2.0':
                s1 = m_s0[b].mean()
                s2 = m_c0[b].mean()
            elif stat == 'scmr_cond@0.5':
                h1 = m_s0[b] & (d_s0[b] <= 0.5)
                h2 = m_c0[b] & (d_c0[b] <= 0.5)
                s1 = h1.sum() / max(m_s0[b].sum(), 1)
                s2 = h2.sum() / max(m_c0[b].sum(), 1)
            elif stat == 'mce@2m':
                both = m_s0[b] & m_c0[b]
                if both.sum() < 5:
                    continue
                s1 = d_s0[b][both].mean()
                s2 = d_c0[b][both].mean()
            deltas.append(s1 - s2)
        deltas = np.array(deltas)
        return (float(np.mean(deltas)), float(np.percentile(deltas, 2.5)),
                float(np.percentile(deltas, 97.5)))

    emit(f'{"group":<10}{"stat":<14}{"delta":>9}{"CI_low":>9}'
         f'{"CI_high":>9}')
    for g in ['Low', 'Mid', 'High']:
        sel = group == g
        for stat in ['match@0.5', 'match@1.0', 'match@2.0',
                     'scmr_cond@0.5', 'mce@2m']:
            r = boot_delta(sel, stat)
            if r is None:
                emit(f'{g:<10}{stat:<14}  (low support)')
                continue
            emit(f'{g:<10}{stat:<14}{r[0]:>+9.4f}{r[1]:>+9.4f}'
                 f'{r[2]:>+9.4f}')

    with open(args.out_prefix + '_report.txt', 'w') as f:
        f.write('\n'.join(lines))
    print(f'\nreport saved to {args.out_prefix}_report.txt')


if __name__ == '__main__':
    main()
