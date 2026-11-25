"""LaTeX rows summarising IIDM vs gradient reversal across conditions with room to remove.

For every (condition, method): best strongest-probe removal over the method's own lambda grid,
rho_best = (p0 - p_min) / (p0 - chance), with p0 the gradient-reversal lambda=0 strongest probe of that
condition, the lambda at which it occurs, the strongest probe there, and the task (identity / activity)
accuracy on the seen cell at that lambda against the lambda=0 value."""
import glob
import pickle

import numpy as np


def load(d):
    """one directory, or several joined by '+' whose lambda points are merged (later directories add points)"""
    out = {}
    for part in d.split('+'):
        f = glob.glob(part + '/*.pkl')
        if f:
            out.update(pickle.load(open(f[0], 'rb')))
    return out or None


def lam(k):
    return float(k.split('=')[1])


def strongest(v, factor):
    P = [np.asarray(v[f'probe_{factor}{s}_tc_seen_seen_raw'], float) for s in ('', '_mlp', '_knn')]
    return np.nanmax(np.stack(P), axis=0)


D = 'gait_experiment_output_v3_'
CONDS = [
    # label, factor slot, chance, baseline dir, [(method label, dir), ...]
    ('MotionSense, subject', 'activity', 1 / 19, D + 'subject_nuisance_discheldout',
     [('GRL', D + 'subject_nuisance_discheldout'),
      ('GRL, + lambda 0.001, 0.01', D + 'subject_nuisance_discheldout+' + D + 'budget_grl_ms'),
      ('IIDM simplified', D + 'iidmn_ms_simplified'),
      ('IIDM faithful', D + 'iidmn_ms_faithful'),
      ('IIDM faithful, 25 ep. + 200 ep.', D + 'iidmn_ms_faithful+' + D + 'budget_ms_faithful200'),
      ('IIDM faithful, 200 ep.', D + 'budget_ms_faithful200')]),
    ('HHAR, device', 'device', 1 / 2, D + 'disc_heldout',
     [('GRL', D + 'disc_heldout'),
      ('GRL, + lambda 0.001, 0.01', D + 'disc_heldout+' + D + 'budget_grl_hhar'),
      ('IIDM simplified, ann.', D + 'iidm_annealed'),
      ('IIDM simplified, const.', D + 'iidm_constant'), ('IIDM faithful', D + 'iidmn_hhar_faithful'),
      ('IIDM faithful, 100 ep.', D + 'iidmn_hhar_long'), ('IIDM faithful, 200 ep.', D + 'budget_hhar_faithful200')]),
    ('HHAR, activity', 'activity', 1 / 3, D + 'disc_heldout',
     [('GRL', D + 'disc_heldout'),
      ('GRL, + lambda 0.001, 0.01', D + 'disc_heldout+' + D + 'budget_grl_hhar'),
      ('IIDM simplified, ann.', D + 'iidm_annealed'),
      ('IIDM simplified, const.', D + 'iidm_constant'), ('IIDM faithful', D + 'iidmn_hhar_faithful'),
      ('IIDM faithful, 100 ep.', D + 'iidmn_hhar_long'), ('IIDM faithful, 200 ep.', D + 'budget_hhar_faithful200')]),
    ('RealWorld, activity', 'activity', 1 / 3, D + 'restart2_realworld_base',
     [('GRL', D + 'restart2_realworld_base'),
      ('GRL, + lambda 0.01', D + 'restart2_realworld_base+' + D + 'budget_grl_rw'),
      ('IIDM faithful', D + 'iidmn_rw_faithful')]),
    ('RealWorld, position', 'device', 1 / 6, D + 'restart2_realworld_base',
     [('GRL', D + 'restart2_realworld_base'),
      ('GRL, + lambda 0.01', D + 'restart2_realworld_base+' + D + 'budget_grl_rw'),
      ('IIDM faithful', D + 'iidmn_rw_faithful')]),
]

for label, factor, chance, based, methods in CONDS:
    base = load(based)
    p0 = float(np.mean(strongest(base['lambda=0'], factor)))
    task0 = base['lambda=0']['cells']['seen_seen']['window_acc'][0]
    print(f"\n{label}: p0={p0:.3f} chance={chance:.3f} task(seen) at lambda=0 = {task0:.3f}")
    for mname, d in methods:
        r = load(d)
        if r is None:
            print(f"  {mname:26s} (not available)")
            continue
        rows = []
        for k in sorted(r, key=lam):
            if lam(k) == 0:
                continue
            st = float(np.mean(strongest(r[k], factor)))
            rows.append((lam(k), st, 100 * (p0 - st) / (p0 - chance), r[k]['cells']['seen_seen']['window_acc'][0]))
        if not rows:
            continue
        b = max(rows, key=lambda x: x[2])
        print(f"  {mname:26s} grid {[x[0] for x in rows]}  best rho={b[2]:5.1f}% at lambda={b[0]:g}  "
              f"strongest probe {b[1]:.3f}  task(seen) {b[3]:.3f}")
        print(f"      LaTeX: {mname} & ${b[2]:.1f}$ & {b[0]:g} & ${b[1]:.3f}$ & ${b[3]:.3f}$ \\\\")
