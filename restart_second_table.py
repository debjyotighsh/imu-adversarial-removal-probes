"""LaTeX rows for the second-dataset restart table (output of run_restart_second_dataset.ps1)."""
import glob
import pickle

import numpy as np

import frozen_analysis as FA

SETS = [('MobiAct B', 'mobiact', ['0', '1', '2']), ('RealWorld', 'realworld', ['0', '1', '2'])]


def load(tag, m):
    return pickle.load(open(glob.glob(f'gait_experiment_output_v3_restart2_{tag}_{m}/*.pkl')[0], 'rb'))


def ms(x):
    x = np.asarray(x, float)
    return x.mean(), x.std(ddof=1)


maxsd = 0
for name, tag, lams in SETS:
    b, r = load(tag, 'base'), load(tag, 'restart')
    for lam in lams:
        k = f'lambda={lam}'
        cells = []
        for res in (b, r):
            v = res[k]
            cells.append(ms(v['disc_act_heldout_tc_seen_seen_raw']))
        for s in ('', '_mlp'):
            for res in (b, r):
                m, sd = ms(res[k][f'probe_activity{s}_tc_seen_seen_raw'])
                maxsd = max(maxsd, sd)
                cells.append((m, sd))
        ent = np.mean([FA.conf_stats(cf)[2] for cf in r[k]['disc_act_conf_raw']])
        ent_b = np.mean([FA.conf_stats(cf)[2] for cf in b[k]['disc_act_conf_raw']])
        d = [f"${m:.3f}$ & ${sd:.3f}$" for m, sd in cells[:2]]
        p = [f"${m:.3f}$" for m, sd in cells[2:]]
        print(f"{name} & {float(lam):g} & {d[0]} & {d[1]} & {ent_b:.2f} & {ent:.2f} & " + " & ".join(p) + r" \\")
print('max probe SD', round(maxsd, 3))
