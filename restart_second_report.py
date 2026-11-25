"""Compare no-restart and restart runs on the second datasets (output of run_restart_second_dataset.ps1)."""
import glob
import pickle

import numpy as np

import frozen_analysis as FA

SETS = {'MobiAct B': ('mobiact_base', 'mobiact_restart'),
        'RealWorld (capped)': ('realworld_base', 'realworld_restart')}


def load(tag):
    p = glob.glob(f'gait_experiment_output_v3_restart2_{tag}/*.pkl')
    return pickle.load(open(p[0], 'rb')) if p else None


def pm(vals, nd=3):
    v = np.asarray(vals, float)
    return f"{np.nanmean(v):.{nd}f}±{(np.nanstd(v, ddof=1) if len(v) > 1 else 0.0):.{nd}f}"


def lam(k):
    return float(k.split('=')[1])


for name, (b, r) in SETS.items():
    rb, rr = load(b), load(r)
    print(f"\n=== {name} ===")
    if rb is None or rr is None:
        print("  (not finished)")
        continue
    print(f"{'lambda':>7} | {'NO RESTART: disc held-out bal':>30} {'entropy':>8} {'top':>5} {'probe':>13} {'gap':>6} |"
          f" {'RESTART: disc held-out bal':>27} {'entropy':>8} {'top':>5} {'probe':>13} {'gap':>6}")
    for k in sorted(rb, key=lam):
        row = []
        for res in (rb, rr):
            v = res[k]
            held = v['disc_act_heldout_tc_seen_seen_raw']
            probe = v['probe_activity_tc_seen_seen_raw']
            st = [FA.conf_stats(cf) for cf in v['disc_act_conf_raw']]
            ent = np.mean([s[2] for s in st])
            top = np.mean([s[3] for s in st])
            gap = 100 * (np.mean(probe) - np.mean(held))
            row.append(f"{pm(held):>30} {ent:8.2f} {top:5.2f} {pm(probe):>13} {gap:6.1f}")
        print(f"{lam(k):7g} | {row[0]} | {row[1]}")
