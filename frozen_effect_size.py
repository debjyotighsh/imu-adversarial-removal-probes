"""Effect-size diagnostics for the discriminator (balanced accuracy, entropy, majority-class
accuracy, confusion matrix, macro-F1), computed from stored confusion counts. No retraining."""
import pickle

import numpy as np
from sklearn.metrics import f1_score

import frozen_analysis as FA
from probe_families_table import find_pkl

D = 'gait_experiment_output_v3_frozen'


def macro_f1_from_conf(cf):
    cf = np.asarray(cf)
    K = cf.shape[0]
    y_true, y_pred = [], []
    for i in range(K):
        for j in range(K):
            n = int(cf[i, j])
            y_true += [i] * n
            y_pred += [j] * n
    return f1_score(y_true, y_pred, average='macro', labels=list(range(K)), zero_division=0)


def report(label, conf_list, chance, class_names=None):
    print(f"\n{label}  (chance {chance:.3f})")
    accs, bals, ents, f1s = [], [], [], []
    for s, cf in enumerate(conf_list):
        acc, bal, ent, top = FA.conf_stats(cf)
        f1 = macro_f1_from_conf(cf)
        accs.append(acc); bals.append(bal); ents.append(ent); f1s.append(f1)
        print(f"  seed {s}: acc {acc:.3f}  balanced acc {bal:.3f}  entropy {ent:.2f}  "
              f"top-share {top:.2f}  macro-F1 {f1:.3f}")
    mean_cf = np.mean([np.asarray(c) for c in conf_list], axis=0)
    names = class_names or [f'c{i}' for i in range(mean_cf.shape[0])]
    print(f"  mean confusion matrix (rows=true, cols=pred), row-% :")
    print(f"           " + "  ".join(f"{n:>10s}" for n in names))
    for i, row in enumerate((100 * mean_cf / mean_cf.sum(1, keepdims=True)).round(1)):
        print(f"  {names[i]:>8s} " + "  ".join(f"{x:9.1f}%" for x in row))
    print(f"  mean over seeds: acc {np.mean(accs):.3f}  balanced acc {np.mean(bals):.3f}  "
          f"entropy {np.mean(ents):.2f}  macro-F1 {np.mean(f1s):.3f} +- {np.std(f1s, ddof=1):.3f}")


r = pickle.load(open(find_pkl(D, 'windows_hz50_win2.56_ov0.5_bike', lambda b: '_cap' not in b and 'dlm5' in b and '_deg' not in b)[0], 'rb'))
report('HHAR activity discriminator, lambda=1 (flagship case)', r['lambda=1']['disc_act_conf_raw'],
       1 / 3, ['bike', 'stairsdn', 'stairsup'])
report('HHAR activity discriminator, lambda=0 (baseline, for contrast)', r['lambda=0']['disc_act_conf_raw'],
       1 / 3, ['bike', 'stairsdn', 'stairsup'])

# also the four non-collapsed near-chance runs from sec:collapse, for completeness
mb = pickle.load(open(find_pkl(D, 'results_sweep_mobiact', lambda b: '_cap' not in b)[0], 'rb'))
for k, s_idx in (('lambda=1', 0), ('lambda=1', 1), ('lambda=2', 2)):
    cf = mb[k]['disc_act_conf_raw'][s_idx]
    f1 = macro_f1_from_conf(cf)
    acc, bal, ent, top = FA.conf_stats(cf)
    print(f"\nMobiAct stairs {k} seed {s_idx}: balanced acc {bal:.3f}  entropy {ent:.2f}  macro-F1 {f1:.3f}")

rr = pickle.load(open(find_pkl(D, 'windows_hz50_win2.56_ov0.5_bike', lambda b: '_cap' not in b and 'dlm5' in b and '_deg' not in b)[0], 'rb'))
cf = rr['lambda=0.05']['disc_dev_conf_raw'][0]
f1 = macro_f1_from_conf(cf)
acc, bal, ent, top = FA.conf_stats(cf)
print(f"\nHHAR device lambda=0.05 seed 0: balanced acc {bal:.3f}  entropy {ent:.2f}  macro-F1 {f1:.3f}")
