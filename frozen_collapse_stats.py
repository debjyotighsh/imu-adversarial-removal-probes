"""Resolves the TODOs in sec:collapse: near-chance count, collapsed count, threshold sensitivity,
the MobiAct-stairs run identities, and the HHAR-device candidate."""
import glob
import pickle

import numpy as np

import frozen_analysis as FA
from probe_families_table import CONDITIONS, find_pkl

D = 'gait_experiment_output_v3_frozen'


def all_runs():
    """(label, factor, lambda, seed, bal_acc, chance, entropy, n_classes, top_share)"""
    out = []
    for label, frag, pred, factors in CONDITIONS:
        hits = find_pkl(D, frag, pred)
        if not hits:
            continue
        r = pickle.load(open(hits[0], 'rb'))
        for fname, key in (('activity', 'disc_act_conf_raw'), ('device/position', 'disc_dev_conf_raw')):
            if fname != 'activity' and not any(k in factors for k in ('device', 'position')):
                continue
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
                    out.append((label, fname, lam, s, bal, 1 / K, ent, K, top))
    return out


rows = all_runs()
print(f"total discriminator runs (lambda>0): {len(rows)}")

for tol in (0.05, 0.10):
    near = [r for r in rows if r[4] - r[5] < tol]
    for ecut, tag in ((0.2, 'collapsed'), ):
        coll = [r for r in near if r[6] < ecut]
        print(f"tol={tol}: N={len(near)} near chance (bal.acc within {tol}), "
              f"M={len(coll)} of them {tag} (entropy<{ecut})  ({100*len(coll)/max(len(near),1):.0f}%)")

print("\n--- non-collapsed-at-chance runs at several (tol, ent_min) settings ---")
for tol, ent_min in ((0.05, 0.9), (0.10, 0.8)):
    sel = [r for r in rows if r[4] - r[5] < tol and r[6] >= ent_min]
    print(f"\ntol={tol}, ent_min={ent_min}: {len(sel)} runs")
    for label, fname, lam, s, bal, c, ent, K, top in sel:
        print(f"  {label:<30}{fname:<16}lambda={lam:<5g} seed {s}: bal.acc {bal:.3f} (chance {c:.3f}) "
              f"entropy {ent:.2f} top-share {top:.2f}")

print("\n--- HHAR device factor: class count and chance ---")
r = pickle.load(open(find_pkl(D, 'windows_hz50_win2.56_ov0.5_bike', lambda b: '_cap' not in b and 'dlm5' in b)[0], 'rb'))
K = np.asarray(r['lambda=0.05']['disc_dev_conf_raw'][0]).shape[0]
print(f"HHAR device discriminator: {K} classes, chance {1/K:.3f}")
for s, cf in enumerate(r['lambda=0.05']['disc_dev_conf_raw']):
    acc, bal, ent, top = FA.conf_stats(cf)
    print(f"  lambda=0.05 seed {s}: bal.acc {bal:.3f} entropy {ent:.2f} top-share {top:.2f}")
