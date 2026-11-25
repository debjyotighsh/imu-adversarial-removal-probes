import glob
import pickle

import numpy as np

p = glob.glob('gait_experiment_output_v3_disc_heldout/*.pkl')[0]
r = pickle.load(open(p, 'rb'))
keys = sorted(r.keys(), key=lambda k: float(k.split('=')[1]))


def pm(vals, nd=3):
    v = np.asarray(vals, float)
    m, s = np.nanmean(v), np.nanstd(v, ddof=1)
    return f"{m:.{nd}f}±{s:.{nd}f}"


print(f"{'lambda':>7} {'disc(train)':>13} {'disc(held-out)':>15} {'probe(held-out)':>16} "
      f"{'gap, held-out disc':>20}")
for k in keys:
    lam = float(k.split('=')[1])
    v = r[k]
    dt = v['disc_act_acc_raw']
    dh = v['disc_act_heldout_tc_seen_seen_raw']
    ph = v['probe_activity_tc_seen_seen_raw']
    gap = 100 * (np.mean(ph) - np.mean(dh))
    print(f"{lam:7g} {pm(dt):>13} {pm(dh):>15} {pm(ph):>16} {gap:19.1f} pts")

print()
print("Per-seed at lambda=1 (the central comparison point):")
v1 = r['lambda=1']
print("  disc (train acc):     ", np.round(v1['disc_act_acc_raw'], 4))
print("  disc (held-out):      ", np.round(v1['disc_act_heldout_tc_seen_seen_raw'], 4))
print("  probe (held-out):     ", np.round(v1['probe_activity_tc_seen_seen_raw'], 4))
print("  chance = 0.333, majority-class rate = 0.373")

print()
print("Held-out discriminator raw correct/n at lambda=1 (for a binomial check against 0.333):")
print("  correct:", v1['disc_act_heldout_tc_seen_seen_correct_raw'])
print("  n:      ", v1['disc_act_heldout_tc_seen_seen_n_raw'])
