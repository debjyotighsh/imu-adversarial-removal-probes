"""IIDM (Iwasawa et al. 2020) vs gradient reversal on the central HHAR condition, same protocol.

Reads gait_experiment_output_v3_iidm_{annealed,constant} (run_iidm.ps1) and, for gradient reversal, the
five-seed run_disc_heldout.ps1 output (first three seeds, to match)."""
import glob
import pickle

import numpy as np

import frozen_analysis as FA

RUNS = [('gradient reversal (GRL)', 'gait_experiment_output_v3_disc_heldout'),
        ('IIDM, lambda annealed', 'gait_experiment_output_v3_iidm_annealed'),
        ('IIDM, lambda constant', 'gait_experiment_output_v3_iidm_constant')]


def pm(v, nd=3):
    v = np.asarray(v, float)
    return f"{v.mean():.{nd}f}±{(v.std(ddof=1) if len(v) > 1 else 0):.{nd}f}"


def load(d):
    f = glob.glob(d + '/*.pkl')
    return pickle.load(open(f[0], 'rb')) if f else None


res = {name: load(d) for name, d in RUNS}
lams = sorted({float(k.split('=')[1]) for r in res.values() if r for k in r})
n = 3
for lam in lams:
    print(f"\n--- lambda={lam:g} (first three seeds) ---")
    print(f"{'method':<26} {'disc held-out':>14} {'entropy':>8} {'lin':>7} {'MLP':>7} {'kNN':>7} {'gap(lin)':>9} "
          f"{'gap(MLP)':>9}  identity seen / unseen-dev")
    for name, _ in RUNS:
        r = res[name]
        key = f"lambda={lam:g}"
        if r is None or key not in r:
            print(f"{name:<26} (not available)")
            continue
        v = r[key]
        held = np.asarray(v['disc_act_heldout_tc_seen_seen_raw'])[:n]
        lin = np.asarray(v['probe_activity_tc_seen_seen_raw'])[:n]
        mlp = np.asarray(v['probe_activity_mlp_tc_seen_seen_raw'])[:n]
        knn = np.asarray(v['probe_activity_knn_tc_seen_seen_raw'])[:n]
        ent = np.mean([FA.conf_stats(cf)[2] for cf in v['disc_act_conf_raw'][:n]])
        idv = (f"{v['cells']['seen_seen']['window_acc'][0]:.3f} / "
               f"{v['cells']['unseen_dev_seen_act']['window_acc'][0]:.3f}")
        print(f"{name:<26} {pm(held):>14} {ent:8.2f} {lin.mean():7.3f} {mlp.mean():7.3f} {knn.mean():7.3f} "
              f"{100 * (lin.mean() - held.mean()):9.1f} {100 * (mlp.mean() - held.mean()):9.1f}  {idv}")
