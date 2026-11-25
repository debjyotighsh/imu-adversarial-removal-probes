"""Table "discheldout" (Section: A Fairer Comparison): raw and balanced discriminator accuracy on
training data, balanced accuracy on the held-out seen/seen cell, and the held-out probe, from one
rerun of the central HHAR sweep (run_disc_heldout.ps1). Every column comes from the same models.

    python disc_heldout_table.py [--dir gait_experiment_output_v3_disc_heldout]
"""
import argparse
import glob
import pickle

import numpy as np

import frozen_analysis as FA


def pm(vals, nd=3):
    v = np.asarray(vals, float)
    return f"{np.nanmean(v):.{nd}f}±{np.nanstd(v, ddof=1):.{nd}f}"


ap = argparse.ArgumentParser()
ap.add_argument('--dir', default='gait_experiment_output_v3_disc_heldout')
a = ap.parse_args()
r = pickle.load(open(glob.glob(f'{a.dir}/*.pkl')[0], 'rb'))
print(f"{'lambda':>7} {'train raw':>14} {'train bal.':>14} {'held-out bal.':>14} {'probe':>14}  gap (train vs held-out), pts")
for k in sorted(r, key=lambda k: float(k.split('=')[1])):
    v = r[k]
    bal_train = [FA.conf_stats(cf)[1] for cf in v['disc_act_conf_raw']]
    held = v['disc_act_heldout_tc_seen_seen_raw']
    probe = v['probe_activity_tc_seen_seen_raw']
    g1 = 100 * (np.mean(probe) - np.mean(bal_train))
    g2 = 100 * (np.mean(probe) - np.mean(held))
    print(f"{float(k.split('=')[1]):7g} {pm(v['disc_act_acc_raw']):>14} {pm(bal_train):>14} {pm(held):>14} "
          f"{pm(probe):>14}  {g1:.1f} vs {g2:.1f}")
