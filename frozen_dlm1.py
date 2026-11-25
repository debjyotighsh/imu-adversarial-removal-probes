"""HHAR at the standard discriminator learning rate (multiplier 1) vs the paper's multiplier 5."""
import glob
import pickle

import numpy as np

import frozen_analysis as FA

for tag in ('dlm5', 'dlm1'):
    p = [x for x in glob.glob('gait_experiment_output_v3_frozen/*bike*.pkl')
         if tag in x and 'segdetail' not in x and 'dynamics' not in x and '_deg' not in x][0]
    r = pickle.load(open(p, 'rb'))
    print(f"\nHHAR {tag}: lambda | disc acc | bal.acc | entropy | top-share | probe in-pool | probe held-out | MLP held-out")
    for k in sorted(r, key=lambda k: float(k.split('=')[1])):
        v = r[k]
        st = np.array([FA.conf_stats(c) for c in v['disc_act_conf_raw']])
        print(f"{k:12s} {FA.ms(v['disc_act_acc_raw'])} {st[:, 1].mean():.3f} {st[:, 2].mean():.2f} {st[:, 3].mean():.2f} | "
              f"{FA.ms(v['probe_activity_raw'])} | {FA.ms(v['probe_activity_tc_seen_seen_raw'])} | "
              f"{FA.ms(v['probe_activity_mlp_tc_seen_seen_raw'])}")
