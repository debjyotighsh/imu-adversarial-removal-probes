"""MotionSense controlled-degradation numbers (held-out probe) from the frozen rerun."""
import glob
import os
import pickle
import re

import numpy as np
from scipy import stats


def load_grid(r, stem, tc, lams):
    fam = {}
    for suf, name in (('', 'lin'), ('_mlp', 'mlp'), ('_knn', 'knn')):
        rows = []
        for l in lams:
            raw = r[f'lambda={l:g}'].get(f'{stem}{suf}_tc_{tc}_raw')
            rows.append(raw if raw is not None else [np.nan] * 3)
        fam[name] = np.array(rows, float).T
    return fam


def rho_seed(p, c):
    p0 = p[:, 0]
    return (p0 - np.nanmin(p, axis=1)) / (p0 - c)


D = 'gait_experiment_output_v3_frozen'
lams = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
c = 1 / 3
b0 = pickle.load(open(
    [p for p in glob.glob(os.path.join(D, '*.pkl'))
     if 'motionsense' in p and '_cap' not in p and '_deg' not in p
     and 'segdetail' not in p and 'dynamics' not in p][0], 'rb'))
rows = [('none', 0.0, b0)]
for p in sorted(glob.glob(os.path.join(D, '*motionsense*deg*.pkl'))):
    if 'segdetail' in p or 'dynamics' in p:
        continue
    m = re.search(r'_deg(noise|mix)([0-9.]+)\.pkl$', p)
    rows.append((m.group(1), float(m.group(2)), pickle.load(open(p, 'rb'))))

pts = []
for mode, lev, r in rows:
    fam = load_grid(r, 'probe_activity', 'seen_seen', lams)
    best = np.nanmax(np.stack(list(fam.values())), axis=0)
    p0 = fam['lin'][:, 0]
    valid = float(np.mean(p0)) - c >= 0.05
    lab = 'none' if mode == 'none' else f'{mode} {lev:g}'
    if not valid:
        print(f'{lab:<10}: p0 {np.round(p0,3)} -- rho undefined')
        continue
    rl = rho_seed(fam['lin'], c)
    rb = rho_seed(best, c)
    print(f'{lab:<10} p0={np.mean(p0):.3f}+-{np.std(p0,ddof=1):.3f}  '
          f'rho_lin={100*np.mean(rl):.1f}+-{100*np.std(rl,ddof=1):.1f}  '
          f'rho_best={100*np.mean(rb):.1f}+-{100*np.std(rb,ddof=1):.1f}')
    pts.append((np.mean(p0), np.mean(rl), np.mean(rb)))

x = [p[0] for p in pts]
print('\nSpearman n=%d: p0 vs rho_lin' % len(pts), round(stats.spearmanr(x, [p[1] for p in pts]).statistic, 2),
      '| p0 vs rho_best', round(stats.spearmanr(x, [p[2] for p in pts]).statistic, 2))
