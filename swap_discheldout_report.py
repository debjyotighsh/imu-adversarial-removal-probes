import pickle

import numpy as np

import frozen_analysis as FA

r = pickle.load(open('gait_experiment_output_v3_subject_nuisance_discheldout/results_subject_nuisance_motionsense.pkl', 'rb'))
keys = sorted(r.keys(), key=lambda k: float(k.split('=')[1]))


def pm(vals, nd=3):
    v = np.asarray(vals, float)
    m, s = np.nanmean(v), np.nanstd(v, ddof=1) if len(v) > 1 else 0.0
    return f"{m:.{nd}f}±{s:.{nd}f}"


print(f"{'lambda':>7} {'disc(train,raw)':>16} {'disc(train,bal)':>16} {'disc(held-out,bal)':>19} "
      f"{'probe(held-out)':>16} {'gap: train-bal vs held-out-bal':>32}")
for k in keys:
    lam = float(k.split('=')[1])
    v = r[k]
    raw_train = v['disc_act_acc_raw']
    confs = v.get('disc_act_conf_raw')
    if confs:
        bal_train = [FA.conf_stats(cf)[1] for cf in confs]
    else:
        bal_train = [float('nan')] * len(raw_train)
    held_bal = v['disc_act_heldout_tc_seen_seen_raw']
    ph = v['probe_activity_tc_seen_seen_raw']
    gap_trainbal = 100 * (np.mean(ph) - np.mean(bal_train))
    gap_heldbal = 100 * (np.mean(ph) - np.mean(held_bal))
    print(f"{lam:7g} {pm(raw_train):>16} {pm(bal_train):>16} {pm(held_bal):>19} {pm(ph):>16} "
          f"  {gap_trainbal:5.1f} vs {gap_heldbal:5.1f} pts")

print()
print("Identity metrics (unseen_subj, seen_seen) and per-seed detail at lambda=1:")
v1 = r['lambda=1']
print("  in-dist activity:", v1['cells']['seen_seen']['bout_acc'])
print("  unseen-subj activity:", v1['cells']['unseen_subj']['bout_acc'])
print("  held-out disc raw per seed:", v1['disc_act_heldout_tc_seen_seen_raw'])
print("  held-out disc n:", v1['disc_act_heldout_tc_seen_seen_n_raw'])
print("  held-out disc correct:", v1['disc_act_heldout_tc_seen_seen_correct_raw'])
