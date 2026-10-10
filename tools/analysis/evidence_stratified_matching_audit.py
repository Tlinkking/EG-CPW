"""Matching-threshold definition audit: C0 vs S0.

Reuses the existing prediction dumps and the shared evidence/group
assignment (identical to evidence_stratified_analysis.py). No model,
loss, pipeline, config, checkpoint or dump file is modified.

Definitions
-----------
Old Match@tau (previous report):
    greedy one-to-one matching with distance cap 2.0 m; a GT counts as
    matched@tau iff it was matched under the 2.0 m assignment AND the
    assigned prediction has d <= tau. This can let a higher-score but
    farther prediction "occupy" a GT and block a closer lower-score one.

StrictMatch@tau (threshold-independent):
    for each tau, the greedy one-to-one matching is re-run with distance
    cap tau itself: predictions in descending score order, each matching
    the nearest unmatched same-class GT with d <= tau. No reuse of the
    2.0 m assignment.

SCMR_cond@tau (original definition, unchanged):
    among GTs matched under the 2.0 m association, the fraction whose
    assigned distance d <= tau; tau in {0.5, 1}. It is strictly a
    "conditional center precision given successful 2m association"
    metric, not a standard Match@tau.

mCE@2m = mean d over GTs matched under StrictMatch@2m.

Bootstrap: paired resampling of the C0/S0 delta, 1000 reps, seed 3407.
Two resampling units are reported:
    - GT instance (same as the previous report);
    - frame cluster (resample sample tokens, pool all their GTs), which
      accounts for within-frame prediction/GT matching correlation.
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


def strict_match(dump, gt_rows, cap):
    """Greedy one-to-one class-aware matching with distance cap `cap`."""
    n = len(gt_rows)
    matched = np.zeros(n, dtype=bool)
    dist = np.full(n, np.nan)

    by_sc = defaultdict(list)
    for i, r in enumerate(gt_rows):
        by_sc[(r['token'], r['cls_id'])].append(i)

    preds = defaultdict(list)
    for token, pr in dump['preds'].items():
        keep = pr['scores'] >= SCORE_THR
        for box, score, label in zip(pr['boxes'][keep], pr['scores'][keep],
                                     pr['labels'][keep]):
            preds[(token, int(label))].append(
                (float(score), float(box[0]), float(box[1])))

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
                dist[gt_ids[j]] = d[j]
    return matched, dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='configs/centerpoint/mycenterpoint_nus_evidence_controlled.py')
    parser.add_argument('--c0', required=True)
    parser.add_argument('--s0', required=True)
    parser.add_argument('--out-prefix', required=True)
    args = parser.parse_args()

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
        assert abs(a['x'] - b['x']) < 1e-4 and abs(a['y'] - b['y']) < 1e-4
    n = len(gt_rows)
    print(f'GT tables identical: {n} GTs')

    cls_ids = np.array([r['cls_id'] for r in gt_rows])
    tokens = np.array([r['token'] for r in gt_rows])
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
    print('group sizes:', {g: int((group == g).sum())
                           for g in ['Low', 'Mid', 'High']})

    # strict matching per threshold, both methods
    strict = {}
    for name, dump in [('C0', c0), ('S0', s0)]:
        for tau in TAUS:
            m, d = strict_match(dump, gt_rows, tau)
            strict[(name, tau)] = (m, d)
            print(f'{name} strict@{tau}: matched={m.sum()}')

    # sanity: StrictMatch@2 must equal the old 2m-cap assignment
    m2_c0, d2_c0 = strict[('C0', 2.0)]
    m2_s0, d2_s0 = strict[('S0', 2.0)]

    def metrics(name, sel):
        out = {}
        m2, d2 = strict[(name, 2.0)]
        for tau in TAUS:
            mt, dt = strict[(name, tau)]
            out[f'strict@{tau}'] = (sel & mt).sum() / sel.sum()
            # old definition, renamed AssocMatch: GT matched under the
            # 2m association whose assigned distance is <= tau
            out[f'assoc@{tau}'] = (sel & m2 & (d2 <= tau)).sum() \
                / sel.sum()
        for tau in [0.5, 1.0]:
            # SCMR_cond keeps the original definition unchanged:
            # among GTs matched under the 2m association, the fraction
            # whose assigned distance d <= tau.
            out[f'scmr_cond@{tau}'] = (
                (sel & m2 & (d2 <= tau)).sum()
                / max((sel & m2).sum(), 1))
        md = d2[sel & m2]
        out['mce@2m'] = float(md.mean()) if len(md) else float('nan')
        out['n'] = int(sel.sum())
        return out

    lines = []

    def emit(s=''):
        print(s)
        lines.append(s)

    # verification required by the audit spec: old Match@2m must equal
    # StrictMatch@2m (identical greedy algorithm and 2m cap)
    r0 = metrics('C0', np.ones(n, dtype=bool))
    r1 = metrics('S0', np.ones(n, dtype=bool))
    diff_c0 = abs(r0['assoc@2.0'] - r0['strict@2.0'])
    diff_s0 = abs(r1['assoc@2.0'] - r1['strict@2.0'])
    emit(f'verification |Match@2 - StrictMatch@2|: C0={diff_c0:.2e}, '
         f'S0={diff_s0:.2e} (both must be ~0)')
    assert diff_c0 < 1e-9 and diff_s0 < 1e-9, \
        'old Match@2m != StrictMatch@2m; stop and report'

    def table(title, selectors):
        emit(f'\n=== {title} ===')
        emit(f'{"group":<22}{"m":<5}{"n":>7}'
             f'{"Assoc@.5":>9}{"Strct@.5":>9}'
             f'{"Assoc@1":>9}{"Strct@1":>9}'
             f'{"Match@2":>9}{"Strct@2":>9}'
             f'{"SCMRc@.5":>9}{"SCMRc@1":>9}{"mCE@2m":>9}')
        for name, sel in selectors:
            for mname in ['C0', 'S0']:
                r = metrics(mname, sel)
                if r['n'] == 0:
                    continue
                flag = ' *low*' if r['n'] < LOW_SUPPORT else ''
                emit(f'{name:<22}{mname:<5}{r["n"]:>7}'
                     f'{r["assoc@0.5"]:>9.4f}{r["strict@0.5"]:>9.4f}'
                     f'{r["assoc@1.0"]:>9.4f}{r["strict@1.0"]:>9.4f}'
                     f'{r["assoc@2.0"]:>9.4f}{r["strict@2.0"]:>9.4f}'
                     f'{r["scmr_cond@0.5"]:>9.4f}{r["scmr_cond@1.0"]:>9.4f}'
                     f'{r["mce@2m"]:>9.4f}{flag}')

    all_sel = np.ones(n, dtype=bool)
    selectors = [(g, group == g) for g in ['Low', 'Mid', 'High']]
    selectors.append(('All', all_sel))
    table('OLD vs STRICT MATCHING (evidence groups)', selectors)

    table('FOCUS CLASSES - LOW EVIDENCE',
          [(cname, (group == 'Low')
            & (cls_ids == class_names.index(cname)))
           for cname in FOCUS_CLASSES])

    # delta comparison
    emit('\n=== DELTA (S0-C0): AssocMatch vs StrictMatch ===')
    emit(f'{"group":<10}{"dAsc@.5":>9}{"dStr@.5":>9}{"dAsc@1":>9}'
         f'{"dStr@1":>9}{"dM@2":>9}{"dStr@2":>9}{"dSCMRc@.5":>10}'
         f'{"dSCMRc@1":>10}{"dmCE@2m":>9}')
    for name, sel in selectors:
        r0, r1 = metrics('C0', sel), metrics('S0', sel)
        emit(f'{name:<10}'
             f'{r1["assoc@0.5"]-r0["assoc@0.5"]:>+9.4f}'
             f'{r1["strict@0.5"]-r0["strict@0.5"]:>+9.4f}'
             f'{r1["assoc@1.0"]-r0["assoc@1.0"]:>+9.4f}'
             f'{r1["strict@1.0"]-r0["strict@1.0"]:>+9.4f}'
             f'{r1["assoc@2.0"]-r0["assoc@2.0"]:>+9.4f}'
             f'{r1["strict@2.0"]-r0["strict@2.0"]:>+9.4f}'
             f'{r1["scmr_cond@0.5"]-r0["scmr_cond@0.5"]:>+10.4f}'
             f'{r1["scmr_cond@1.0"]-r0["scmr_cond@1.0"]:>+10.4f}'
             f'{r1["mce@2m"]-r0["mce@2m"]:>+9.4f}')

    # ---- bootstrap ----
    uniq_tokens = np.unique(tokens)
    tok2idx = {t: np.where(tokens == t)[0] for t in uniq_tokens}
    rng = np.random.default_rng(BOOT_SEED)

    def stat_vals(idx):
        if len(idx) == 0:
            return None
        vals = {}
        for name in ['C0', 'S0']:
            m2, d2 = strict[(name, 2.0)]
            m05, _ = strict[(name, 0.5)]
            m10, _ = strict[(name, 1.0)]
            md = d2[idx][m2[idx]]
            vals[name] = dict(
                s05=m05[idx].mean(), s10=m10[idx].mean(),
                s20=m2[idx].mean(),
                scmr=((m2[idx] & (d2[idx] <= 0.5)).sum()
                      / max(m2[idx].sum(), 1)),
                mce=float(md.mean()) if len(md) >= 5 else np.nan)
        out = {k: vals['S0'][k] - vals['C0'][k] for k in vals['C0']}
        if np.isnan(out['mce']):
            out.pop('mce')
        return out

    def bootstrap(gname, unit):
        """unit: 'gt' (instance resample) or 'frame' (cluster resample)."""
        if unit == 'gt':
            pool = np.where(group == gname)[0]
        else:
            pool = uniq_tokens
        acc = defaultdict(list)
        for _ in range(N_BOOT):
            if unit == 'gt':
                idx = pool[rng.integers(0, len(pool), len(pool))]
            else:
                tt = pool[rng.integers(0, len(pool), len(pool))]
                idx = np.concatenate([tok2idx[t] for t in tt])
                idx = idx[group[idx] == gname]
            r = stat_vals(idx)
            if r is None:
                continue
            for k, v in r.items():
                acc[k].append(v)
        return {k: (float(np.mean(v)), float(np.percentile(v, 2.5)),
                    float(np.percentile(v, 97.5)))
                for k, v in acc.items()}

    emit(f'\n=== BOOTSTRAP 95% CI of delta (S0-C0), {N_BOOT} paired '
         f'resamples, seed={BOOT_SEED} ===')
    emit('resampling unit: GT instance | frame cluster')
    emit(f'{"group":<6}{"stat":<10}{"d_gt":>24}{"d_frame":>24}')
    for gname in ['Low', 'Mid', 'High']:
        bg = bootstrap(gname, 'gt')
        bf = bootstrap(gname, 'frame')
        for stat, label in [('s05', 'StrictM@.5'), ('s10', 'StrictM@1'),
                            ('s20', 'StrictM@2'), ('scmr', 'SCMRc@.5'),
                            ('mce', 'mCE@2m')]:
            if stat not in bg or stat not in bf:
                continue
            a, b = bg[stat], bf[stat]
            emit(f'{gname:<6}{label:<10}'
                 f'  {a[0]:+.4f} [{a[1]:+.4f},{a[2]:+.4f}]'
                 f'  {b[0]:+.4f} [{b[1]:+.4f},{b[2]:+.4f}]')

    with open(args.out_prefix + '_report.txt', 'w') as f:
        f.write('\n'.join(lines))
    print(f'\nreport saved to {args.out_prefix}_report.txt')


if __name__ == '__main__':
    main()
