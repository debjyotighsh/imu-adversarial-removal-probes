"""Controlled degradation re-estimated with lambda chosen on separate data (output of run_selection_degrade.ps1).

Per level: p0 (linear, whole cell, lambda=0), and the probe-accuracy reduction rho for the linear probe and the strongest
of three families under
  best     minimum over lambda on the whole held-out cell (the estimator of the paper's original table)
  selected lambda chosen per seed on half A of the cell, rho read on half B (see selection_report.py)
Then the Spearman rank correlation between p0 and the mean rho over levels, for noise, class mix and pooled.
Baseline ('none') levels come from run_selection.ps1 (first three seeds; HHAR restricted to the degradation lambda grid)."""
import glob
import pickle
import re

import numpy as np
from scipy.stats import spearmanr

D = 'gait_experiment_output_v3_'
HH_GRID = [0.0, 0.05, 0.1, 0.25, 1.0]


def load(path):
    return pickle.load(open(path, 'rb'))


def curves(r, lams, suffix=''):
    keys = [f'lambda={l:g}' for l in lams]
    out = {}
    for fs, name in (('', 'lin'), ('_mlp', 'mlp'), ('_knn', 'knn')):
        rows = [r[k].get(f'probe_activity{fs}_tc_seen_seen{suffix}_raw', [np.nan] * 3)[:3] for k in keys]
        out[name] = np.array(rows, float).T
    return out


def estimates(r, lams, c):
    full, A, B = curves(r, lams), curves(r, lams, '_halfA'), curves(r, lams, '_halfB')
    res = {'p0': float(np.nanmean(full['lin'][:, 0]))}
    for fam, f in (('lin', lambda P: P['lin']), ('best', lambda P: np.nanmax(np.stack(list(P.values())), axis=0))):
        F, Aa, Bb = f(full), f(A), f(B)
        n = F.shape[0]
        best = np.array([(F[s, 0] - np.nanmin(F[s])) / (F[s, 0] - c) for s in range(n)])
        istar = np.nanargmin(Aa, axis=1)
        sel = np.array([(Bb[s, 0] - Bb[s, istar[s]]) / (Bb[s, 0] - c) for s in range(n)])
        res[fam] = (best, sel)
    return res


def ms(x):
    x = 100 * np.asarray(x, float)
    return f"{np.nanmean(x):5.1f}±{np.nanstd(x, ddof=1):4.1f}"


def run(name, c, base_dir, lams, base_grid, files_glob, levels_order, skip=()):
    print(f"\n=== {name} (chance {c:.3f}) ===")
    rows = []
    b = glob.glob(base_dir + '/*.pkl')
    if b:
        rows.append(('none', 0.0, estimates(load(b[0]), lams, c)))
    for p in sorted(glob.glob(files_glob)):
        m = re.search(r'_deg(noise|mix)([0-9.]+)\.pkl$', p)
        if not m or 'segdetail' in p or 'dynamics' in p:
            continue
        r = load(p)
        if any(f'lambda={l:g}' not in r for l in lams):
            print(f"  (skipping incomplete checkpoint {p.split('_deg')[-1]})")
            continue
        rows.append((m.group(1), float(m.group(2)), estimates(r, lams, c)))
    rows.sort(key=lambda x: (levels_order.get(x[0], 9), x[1]))
    print(f"{'degradation':<12} {'p0':>6} | {'lin best':>11} {'lin sel':>11} | {'strongest best':>14} {'strongest sel':>14}")
    for mode, lev, e in rows:
        print(f"{mode + ' ' + (str(lev) if mode != 'none' else ''):<12} {e['p0']:6.3f} | {ms(e['lin'][0]):>11} {ms(e['lin'][1]):>11} | "
              f"{ms(e['best'][0]):>14} {ms(e['best'][1]):>14}")
    valid = [r for r in rows if r[2]['p0'] - c >= 0.05 and (r[0], r[1]) not in skip]
    for label, sel in (('noise', [r for r in valid if r[0] in ('none', 'noise')]),
                       ('mix', [r for r in valid if r[0] in ('none', 'mix')]),
                       ('pooled', valid)):
        if len(sel) < 3:
            continue
        p0 = [r[2]['p0'] for r in sel]
        out = []
        for fam in ('lin', 'best'):
            for k, kn in ((0, 'best'), (1, 'sel')):
                rho = [float(np.nanmean(r[2][fam][k])) for r in sel]
                out.append(f"{fam}/{kn} {spearmanr(p0, rho).correlation:+.2f}")
        print(f"  Spearman(p0, rho), {label:6s} n={len(sel)}: " + "   ".join(out))


run('HHAR ambulatory', 1 / 3, D + 'select_hhar', HH_GRID, HH_GRID, D + 'select_degrade/*windows_hz50*deg*.pkl', {'none': 0, 'noise': 1, 'mix': 2})
run('MotionSense ambulatory', 1 / 3, D + 'select_motionsense_amb', [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0], None,
    D + 'select_degrade/*motionsense*deg*.pkl', {'none': 0, 'noise': 1, 'mix': 2})
