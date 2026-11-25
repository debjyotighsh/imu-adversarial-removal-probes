import glob
import pickle

import numpy as np

import frozen_analysis as FA

p = glob.glob('gait_experiment_output_v3_disc_restart/*.pkl')[0]
r = pickle.load(open(p, 'rb'))
keys = sorted(r.keys(), key=lambda k: float(k.split('=')[1]))


def pm(vals, nd=3):
    v = np.asarray(vals, float)
    m, s = np.nanmean(v), (np.nanstd(v, ddof=1) if len(v) > 1 else 0.0)
    return f"{m:.{nd}f}±{s:.{nd}f}"


print(f"{'lambda':>7} {'disc(train,raw)':>16} {'disc(train,bal)':>16} {'disc(held-out,bal)':>19} "
      f"{'probe(held-out)':>16} {'gap(held-out) pts':>18}")
for k in keys:
    lam = float(k.split('=')[1])
    v = r[k]
    raw_train = v['disc_act_acc_raw']
    confs = v.get('disc_act_conf_raw')
    bal_train = [FA.conf_stats(cf)[1] for cf in confs]
    held_bal = v['disc_act_heldout_tc_seen_seen_raw']
    ph = v['probe_activity_tc_seen_seen_raw']
    gap = 100 * (np.mean(ph) - np.mean(held_bal))
    print(f"{lam:7g} {pm(raw_train):>16} {pm(bal_train):>16} {pm(held_bal):>19} {pm(ph):>16} {gap:17.1f}")

print()
print("Per-seed at lambda=1:")
v1 = r['lambda=1']
print("  disc held-out bal:", np.round(v1['disc_act_heldout_tc_seen_seen_raw'], 3))
print("  probe held-out:   ", np.round(v1['probe_activity_tc_seen_seen_raw'], 3))
confs = v1['disc_act_conf_raw']
for s, cf in enumerate(confs):
    acc, bal, ent, top = FA.conf_stats(cf)
    print(f"  seed {s}: train balanced_acc={bal:.3f}  entropy={ent:.3f}  top_share={top:.3f}")
