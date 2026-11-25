"""
Seed-level spread for every removal statistic
=============================================
Reads the sweeps written with --nonlinear_probe --knn_probe (they keep the raw per-seed
probe accuracies) and reports, per condition and factor, the removal statistics computed
SEPARATELY for each seed and then summarised across seeds:

    mean +/- SD (sample SD, n-1), median [min, max], and the lambda* at which each seed's
    linear probe is lowest (so it is visible whether the same lambda gives the minimum in
    every seed).

Also prints, for the point estimate used so far (rho of the seed-mean curve), the difference
to the mean of per-seed rho: the minimum over lambda is biased by seed noise, and the two
are not the same statistic.

    python seed_stats.py --dir gait_experiment_output_v2_probes
"""
import argparse
import glob
import os
import pickle

import numpy as np

from probe_families_table import CONDITIONS, find_pkl


def seed_curves(results, stem, tc=None):
    """(lambdas, {family: (seeds, lambdas) array}). tc='seen_seen' etc. selects the probes
    that were fitted on the training pool and scored on that evaluation cell."""
    keys = sorted(results, key=lambda k: float(k.split('=')[1]))
    lam = np.array([float(k.split('=')[1]) for k in keys])
    out = {}
    for suffix, name in (('', 'lin'), ('_mlp', 'mlp'), ('_knn', 'knn')):
        rows = []
        for k in keys:
            raw = results[k].get(stem + suffix + (f'_tc_{tc}' if tc else '') + '_raw')
            rows.append(raw if raw is not None else [np.nan])
        arr = np.array(rows, float).T        # (seeds, lambdas)
        if np.isfinite(arr).any():
            out[name] = arr
    return lam, out


def rho_rows(p, c):
    p0 = p[:, 0]
    return (p0 - np.nanmin(p, axis=1)) / (p0 - c)


def summ(x, scale=1.0, nd=1):
    x = np.asarray(x, float) * scale
    return f"{np.nanmean(x):.{nd}f} ± {np.nanstd(x, ddof=1):.{nd}f}"


def analyse(results, stem, c, tc=None):
    lam, P = seed_curves(results, stem, tc)
    n = P['lin'].shape[0]
    best = np.nanmax(np.stack(list(P.values())), axis=0)
    nl = np.nanmax(np.stack([P[k] for k in P if k != 'lin']), axis=0) if len(P) > 1 else None
    istar = np.nanargmin(P['lin'], axis=1)
    res = dict(lam=lam, n=n, p0=P['lin'][:, 0], istar=istar, lam_star=lam[istar])
    for name, arr in list(P.items()) + [('best', best)]:
        res['rho_' + name] = rho_rows(arr, c)
    if nl is not None:
        res['Rnl'] = np.array([(nl[s, istar[s]] - c) / (nl[s, 0] - c) for s in range(n)])
    # point estimate used previously: rho of the seed-mean curve
    mean_best = np.nanmax(np.stack([np.nanmean(v, axis=0) for v in P.values()]), axis=0)
    res['pt_lin'] = (np.nanmean(P['lin'][:, 0]) - np.nanmin(np.nanmean(P['lin'], axis=0))) / (np.nanmean(P['lin'][:, 0]) - c)
    res['pt_best'] = (mean_best[0] - np.nanmin(mean_best)) / (mean_best[0] - c)
    res['curves'] = P
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='gait_experiment_output_v2_probes')
    args = ap.parse_args()
    for label, frag, pred, factors in CONDITIONS:
        hits = find_pkl(args.dir, frag, pred)
        if not hits:
            print(f"\n{label}: no checkpoint")
            continue
        results = pickle.load(open(hits[0], 'rb'))
        for fname, (stem, c) in factors.items():
            try:
                r = analyse(results, stem, c)
            except Exception as e:                          # noqa: BLE001
                print(f"\n{label} -- {fname}: {type(e).__name__}: {e}")
                continue
            print(f"\n{label} -- {fname}  (n={r['n']} seeds, chance {c:.3f})")
            print(f"  p0            {summ(r['p0'], 1, 3)}")
            for k in ('lin', 'mlp', 'knn', 'best'):
                if 'rho_' + k in r:
                    v = r['rho_' + k] * 100
                    print(f"  rho_{k:<5}   {summ(v)}   median {np.nanmedian(v):.1f} [{np.nanmin(v):.1f}, {np.nanmax(v):.1f}]")
            if 'Rnl' in r:
                print(f"  R_nl          {summ(r['Rnl'], 1, 2)}")
            print(f"  lambda* per seed {list(r['lam_star'])}   seed-mean-curve rho_lin={100*r['pt_lin']:.1f} rho_best={100*r['pt_best']:.1f}")


if __name__ == '__main__':
    main()
