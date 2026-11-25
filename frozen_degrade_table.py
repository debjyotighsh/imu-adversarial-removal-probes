"""Controlled-degradation table from the frozen rerun, held-out probe (probe_activity_tc_seen_seen)."""
import glob
import os
import pickle
import re

import numpy as np

F = np.sqrt(1.5)


def pm(m, s, nd=3):
    s_txt = f"{s:.{nd}f}" if round(s, nd) > 0 else "{<}" + f"{10 ** -nd:.{nd}f}"
    return f"${m:.{nd}f}\\pm{s_txt}$"


def pmv(x, nd=1, scale=100.0):
    x = np.asarray(x, float) * scale
    return pm(np.nanmean(x), np.nanstd(x, ddof=1), nd)


def rho_seed(p, c):
    p0 = p[:, 0]
    return (p0 - np.nanmin(p, axis=1)) / (p0 - c)


def load_grid(r, stem, tc, lams):
    fam = {}
    for suf, name in (('', 'lin'), ('_mlp', 'mlp'), ('_knn', 'knn')):
        rows = []
        for l in lams:
            key = f'{stem}{suf}_tc_{tc}_raw'
            raw = r[f'lambda={l:g}'].get(key)
            rows.append(raw if raw is not None else [np.nan] * 3)
        fam[name] = np.array(rows, float).T
    return fam


D = 'gait_experiment_output_v3_frozen'
b0 = pickle.load(open(
    [p for p in glob.glob(os.path.join(D, '*.pkl'))
     if 'windows_hz50_win2.56_ov0.5_bike' in p and '_cap' not in p and '_deg' not in p
     and 'dlm5' in p and 'segdetail' not in p and 'dynamics' not in p][0], 'rb'))
lams = [0.0, 0.05, 0.1, 0.25, 1.0]
c = 1 / 3
print('%% HHAR ambulatory, held-out probe, controlled degradation (frozen rerun)')
rows = [('none', 0.0, b0)]
for p in sorted(glob.glob(os.path.join(D, '*windows_hz50*deg*.pkl'))):
    if 'segdetail' in p or 'dynamics' in p:
        continue
    m = re.search(r'_deg(noise|mix)([0-9.]+)\.pkl$', p)
    rows.append((m.group(1), float(m.group(2)), pickle.load(open(p, 'rb'))))

for mode, lev, r in rows:
    fam = load_grid(r, 'probe_activity', 'seen_seen', lams)
    best = np.nanmax(np.stack(list(fam.values())), axis=0)
    p0 = fam['lin'][:, 0]
    valid = float(np.mean(p0)) - c >= 0.05
    drop = p0 - np.nanmin(fam['lin'], axis=1)
    drop_best = best[:, 0] - np.nanmin(best, axis=1)
    im, isd = r['lambda=0']['cells']['seen_seen']['bout_acc']
    ident = pm(im, isd * F)
    lab = 'none' if mode == 'none' else f'{mode} {lev:g}'
    if not valid:
        print(f'  {lab:<10}: p0 {pmv(p0,3,1)} -- rho undefined (within 0.05 of chance)')
        continue
    rl = rho_seed(fam['lin'], c)
    rb = rho_seed(best, c)
    print(f'{lab:<10} & {pmv(p0,3,1)} & {pmv(rl)} & {pmv(rb)} & {pmv(drop,3,1)} & {ident} \\\\')
    print(f'  %% raw p0={np.round(p0,4)} lin_rho={np.round(100*rl,1)} best_rho={np.round(100*rb,1)}')
