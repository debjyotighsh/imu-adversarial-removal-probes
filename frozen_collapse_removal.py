"""
For every discriminator run at chance (balanced accuracy within a tolerance of chance), compute
how much of the decodable information the held-out STRONGEST probe shows as removed AT THAT
SAME (lambda, seed) point: rho_point = (p0 - p_lambda) / (p0 - c), using each seed's own lambda=0
baseline and the best of the three held-out probe families at that lambda.

This answers "what does the probe show whenever the discriminator is at chance", which is a
different question from Table 5's rho (a minimum over the whole sweep).
"""
import pickle

import numpy as np

import frozen_analysis as FA
from probe_families_table import CONDITIONS, find_pkl

D = 'gait_experiment_output_v3_frozen'


def stem_for(fname):
    return 'probe_activity' if fname == 'activity' else 'probe_device'


def best_probe_at(r, stem, lam, seed):
    vals = []
    for suf in ('', '_mlp', '_knn'):
        raw = r[f'lambda={lam:g}'].get(f'{stem}{suf}_tc_seen_seen_raw')
        if raw is not None:
            vals.append(raw[seed])
    return max(vals) if vals else np.nan


def collect(tol):
    out = []
    for label, frag, pred, factors in CONDITIONS:
        hits = find_pkl(D, frag, pred)
        if not hits:
            continue
        r = pickle.load(open(hits[0], 'rb'))
        for fname, key in (('activity', 'disc_act_conf_raw'), ('device/position', 'disc_dev_conf_raw')):
            if fname != 'activity' and not any(k in factors for k in ('device', 'position')):
                continue
            stem = stem_for(fname)
            p0_vec = [best_probe_at(r, stem, 0.0, s) for s in range(3)]
            c = None
            for factor_name, (st, cc) in factors.items():
                if st == stem:
                    c = cc
            for k in sorted(r, key=lambda k: float(k.split('=')[1])):
                lam = float(k.split('=')[1])
                if lam == 0:
                    continue
                confs = r[k].get(key)
                if not confs or confs[0] is None:
                    continue
                K = np.asarray(confs[0]).shape[0]
                for s, cf in enumerate(confs):
                    acc, bal, ent, top = FA.conf_stats(cf)
                    if bal - 1 / K >= tol:
                        continue
                    p_lam = best_probe_at(r, stem, lam, s)
                    p0 = p0_vec[s]
                    if not (np.isfinite(p_lam) and np.isfinite(p0)) or p0 - c <= 1e-6:
                        continue
                    rho = (p0 - p_lam) / (p0 - c)
                    out.append((label, fname, lam, s, bal, 1 / K, ent, rho))
    return out


for tol in (0.05, 0.10):
    rows = collect(tol)
    rhos = np.array([r[7] * 100 for r in rows])
    rhos_clip = np.clip(rhos, 0, None)   # a negative point-rho means the probe read ABOVE baseline
    print(f"\ntol={tol}: n={len(rows)} chance-level (condition,factor,lambda,seed) points")
    print(f"  point-rho (strongest held-out probe, vs that seed's own lambda=0): "
          f"median {np.median(rhos):.1f}%  max {rhos.max():.1f}%  min {rhos.min():.1f}%  "
          f"(negative = probe read above its own baseline)")
    print(f"  clipped at 0: median {np.median(rhos_clip):.1f}%  mean {np.mean(rhos_clip):.1f}%")
    if tol == 0.05:
        coll = np.array([r[7] * 100 for r in rows if r[6] < 0.2])
        inter = np.array([r[7] * 100 for r in rows if 0.2 <= r[6] < 0.9])
        spread = np.array([r[7] * 100 for r in rows if r[6] >= 0.9])
        for name, arr in (('collapsed (entropy<0.2)', coll), ('intermediate', inter), ('evenly spread (entropy>=0.9)', spread)):
            if len(arr):
                print(f"    {name}: n={len(arr)}  median {np.median(arr):.1f}%  max {arr.max():.1f}%  min {arr.min():.1f}%")
    if tol == 0.10:
        loose = np.array([r[7] * 100 for r in rows if r[6] >= 0.8])
        print(f"  of these, entropy>=0.8 (the 'non-collapsed' loose set): n={len(loose)}  "
              f"median {np.median(loose):.1f}%  max {loose.max():.1f}%  min {loose.min():.1f}%")

# worst (largest) point-rho overall, to see if ANY chance-level run shows real removal
rows5 = collect(0.05)
rows5.sort(key=lambda r: -r[7])
print("\nlargest point-rho values among the 54 (tol=0.05) runs:")
for label, fname, lam, s, bal, c, ent, rho in rows5[:8]:
    print(f"  {label:<30}{fname:<16}lambda={lam:<5g} seed {s}: bal.acc {bal:.3f} (c {c:.3f}) "
          f"entropy {ent:.2f}  point-rho {100*rho:.1f}%")
