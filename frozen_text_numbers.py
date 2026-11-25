"""Numbers quoted in the prose, from the frozen rerun."""
import glob
import pickle

import numpy as np
from scipy import stats

import frozen_analysis as FA
import seed_stats as SS
from probe_families_table import CONDITIONS, find_pkl

D = 'gait_experiment_output_v3_frozen'


def get(frag, pred):
    return pickle.load(open(find_pkl(D, frag, pred)[0], 'rb'))


# --- HHAR central, lambda = 1 -------------------------------------------------------------
r = get('windows_hz50_win2.56_ov0.5_bike', lambda b: '_cap' not in b and 'dlm5' in b)
v = r['lambda=1']
d = np.array(v['disc_act_acc_raw'])
q = np.array(v['probe_activity_tc_seen_seen_raw'])
qi = np.array(v['probe_activity_raw'])
st = np.array([FA.conf_stats(c) for c in v['disc_act_conf_raw']])
print('HHAR lambda=1  disc per seed', np.round(d, 4), ' probe held-out', np.round(q, 4), ' in-pool', np.round(qi, 4))
print('  gap held-out per seed (pts)', np.round((q - d) * 100, 1), ' mean', round(float(np.mean(q - d) * 100), 1),
      ' | in-pool gap mean', round(float(np.mean(qi - d) * 100), 1))
print('  balanced acc', np.round(st[:, 1], 3), ' entropy', np.round(st[:, 2], 3), ' top-share', np.round(st[:, 3], 4))
# all lambda >= 0.25: is the discriminator a constant predictor?
for k in sorted(r, key=lambda k: float(k.split('=')[1])):
    lam = float(k.split('=')[1])
    if lam >= 0.25:
        s = np.array([FA.conf_stats(c) for c in r[k]['disc_act_conf_raw']])
        print(f'  lambda={lam:g}: top-share {s[:,3].round(3)}  entropy {s[:,2].round(3)}')
p0 = float(np.mean(np.array(r['lambda=0']['probe_activity_tc_seen_seen_raw'])))
print('  held-out probe range across all lambda (seed values):',
      min(min(x['probe_activity_tc_seen_seen_raw']) for x in r.values()),
      max(max(x['probe_activity_tc_seen_seen_raw']) for x in r.values()))
# identity at lambda 0 / 1
print('  identity in-dist', r['lambda=0']['cells']['seen_seen']['bout_acc'], r['lambda=1']['cells']['seen_seen']['bout_acc'])

# --- rank correlation of removal with prior decodability (held-out probes), descriptive -------
rows = []
for label, frag, pred, factors in CONDITIONS:
    rr = get(frag, pred)
    stem, c = factors['activity']
    for tc in (None, 'seen_seen'):
        a = SS.analyse(rr, stem, c, tc=tc)
        rows.append((label, tc, float(np.mean(a['p0'])), float(np.mean(100 * a['rho_lin'])),
                     float(np.mean(100 * a['rho_best']))))
for tc in (None, 'seen_seen'):
    sel = [x for x in rows if x[1] == tc and 'RealWorld (ambulatory, full)' not in x[0]]
    p = [x[2] for x in sel]
    print(f"\nSpearman across {len(sel)} conditions ({'held-out' if tc else 'in-pool'} probes):",
          'p0 vs rho_lin', round(stats.spearmanr(p, [x[3] for x in sel]).statistic, 2),
          '| p0 vs rho_best', round(stats.spearmanr(p, [x[4] for x in sel]).statistic, 2))
    for x in sel:
        print(f"   {x[0]:<34} p0={x[2]:.3f} rho_lin={x[3]:5.1f} rho_best={x[4]:5.1f}")

# --- RealWorld effective band, held-out linear probe ------------------------------------------
for frag, pred, name in (('results_sweep_realworld', lambda b: '_cap' not in b, 'RW full'),
                         ('results_sweep_realworld', lambda b: '_cap15315' in b, 'RW 15.3k')):
    rr = get(frag, pred)
    for fac, stem in (('activity', 'probe_activity'), ('position', 'probe_device')):
        lam, P = SS.seed_curves(rr, stem, 'seen_seen')
        m = np.nanmean(P['lin'], axis=0)
        mid = (m[0] + m.min()) / 2
        print(f"{name} {fac}: p0 {m[0]:.3f} min {m.min():.3f} at lambda {lam[np.argmin(m)]:g}; midpoint {mid:.3f}; "
              f"inside band: {[float(l) for l, x in zip(lam, m) if x < mid]}")
