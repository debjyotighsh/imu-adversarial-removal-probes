"""Per-seed probe values for the four non-collapsed near-chance runs identified above."""
import pickle

import numpy as np

import frozen_analysis as FA
from probe_families_table import find_pkl

D = 'gait_experiment_output_v3_frozen'

r = pickle.load(open(find_pkl(D, 'results_sweep_mobiact', lambda b: '_cap' not in b)[0], 'rb'))
print('MobiAct B (stairs), held-out probe, activity:')
print('  lambda=0 (baseline):', r['lambda=0']['probe_activity_tc_seen_seen_raw'])
print('  lambda=1:', r['lambda=1']['probe_activity_tc_seen_seen_raw'],
      'disc:', r['lambda=1']['disc_act_acc_raw'])
print('  lambda=2:', r['lambda=2']['probe_activity_tc_seen_seen_raw'],
      'disc:', r['lambda=2']['disc_act_acc_raw'])
for k in ('lambda=1', 'lambda=2'):
    for s, cf in enumerate(r[k]['disc_act_conf_raw']):
        acc, bal, ent, top = FA.conf_stats(cf)
        print(f'    {k} seed {s}: bal.acc {bal:.3f} entropy {ent:.3f}')

r = pickle.load(open(find_pkl(D, 'windows_hz50_win2.56_ov0.5_bike', lambda b: '_cap' not in b and 'dlm5' in b)[0], 'rb'))
print('\nHHAR device, held-out probe:')
print('  lambda=0 (baseline):', r['lambda=0']['probe_device_tc_seen_seen_raw'])
print('  lambda=0.05:', r['lambda=0.05']['probe_device_tc_seen_seen_raw'],
      'disc:', r['lambda=0.05']['disc_dev_acc_raw'])
for s, cf in enumerate(r['lambda=0.05']['disc_dev_conf_raw']):
    acc, bal, ent, top = FA.conf_stats(cf)
    print(f'    seed {s}: bal.acc {bal:.3f} entropy {ent:.3f}')
