"""Linear / MLP / k-NN held-out probe readings for the second-dataset restart runs."""
import glob
import pickle

import numpy as np

for tag in ['mobiact', 'realworld']:
    for m in ['base', 'restart']:
        r = pickle.load(open(glob.glob(f'gait_experiment_output_v3_restart2_{tag}_{m}/*.pkl')[0], 'rb'))
        print(tag, m)
        for k in sorted(r, key=lambda k: float(k.split('=')[1])):
            v = r[k]
            vals = [f"{n}={np.mean(v[f'probe_activity{s}_tc_seen_seen_raw']):.3f}"
                    for n, s in [('lin', ''), ('mlp', '_mlp'), ('knn', '_knn')]]
            print('  ', k, *vals)
