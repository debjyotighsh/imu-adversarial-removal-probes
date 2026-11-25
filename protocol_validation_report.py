"""Validation of the evaluation protocol (paper Section "Does the Protocol Add Anything?").

Part 1  Compare three reporting rules on every condition x factor, all with the same thresholds (protocol_rule.py):
   B1  discriminator + linear probe only:  naive best-case linear reduction (minimum over lambda on the whole cell)
   B2  all probe families, no decision rule: naive best-case reduction of the strongest of linear / MLP / k-NN
   P   the protocol: lambda chosen on half A, strongest probe read on half B, with the headroom rule
   Where P (or B2) differs from B1 we check whether a stronger attack supports the change: at the lambda that B1 selected, the
   strongest probe minus the linear probe on half B (per seed) and, where the run has one, on the unseen-device / unseen-activity cell.
Part 2  The three validation conditions (run_protocol_validation.ps1, seeds 10-12, rule frozen in protocol_frozen.txt): the same
   table, plus the predictions P1-P4 written down before the runs.
Part 3  An actionable use case: choosing lambda. Four rules pick one lambda per condition from the seed-mean curves, and the outcome is
   read on data the choice did not use (half B of the seen/seen cell):
      fixed       lambda = 1 (the common default)
      discriminator  the lambda with the lowest training discriminator accuracy (the training-time check alone)
      linear      the lambda with the lowest linear probe on half A
      protocol    the lambda with the lowest strongest-of-three probe on half A; flagged 'no benefit' if the strongest probe's
                  reduction on A is under the 'recoverable' threshold, in which case the recommendation is lambda = 0
   Outcomes: reduction of the strongest probe on half B (higher is better), and the change in in-distribution identity (segment rank-1).
Usage: python protocol_validation_report.py
"""
import glob
import pickle

import numpy as np

import protocol_rule as R

D = 'gait_experiment_output_v3_'
DEV = [  # (condition, dir, factor, probe stem, disc key, chance, unseen cell for the generalisation check)
    ('HHAR', 'select_hhar', 'activity', 'probe_activity', 'disc_act_acc', 1 / 3, 'unseen_dev_seen_act'),
    ('HHAR', 'select_hhar', 'device', 'probe_device', 'disc_dev_acc', 1 / 2, 'seen_dev_unseen_act'),
    ('RealWorld 102.9k', 'select_realworld_full', 'activity', 'probe_activity', 'disc_act_acc', 1 / 3, 'unseen_dev_seen_act'),
    ('RealWorld 102.9k', 'select_realworld_full', 'position', 'probe_device', 'disc_dev_acc', 1 / 6, 'seen_dev_unseen_act'),
    ('RealWorld 15.3k', 'select_realworld_cap', 'activity', 'probe_activity', 'disc_act_acc', 1 / 3, 'unseen_dev_seen_act'),
    ('RealWorld 15.3k', 'select_realworld_cap', 'position', 'probe_device', 'disc_dev_acc', 1 / 6, 'seen_dev_unseen_act'),
    ('MobiAct B', 'select_mobiact_B', 'activity', 'probe_activity', 'disc_act_acc', 1 / 2, None),
    ('MobiAct A', 'select_mobiact_A', 'activity', 'probe_activity', 'disc_act_acc', 1 / 2, None),
    ('MotionSense amb.', 'select_motionsense_amb', 'activity', 'probe_activity', 'disc_act_acc', 1 / 3, None),
    ('MotionSense all', 'select_motionsense_all', 'activity', 'probe_activity', 'disc_act_acc', 1 / 5, None),
]
NEW = [
    ('N1 HHAR all activities', 'validate_hhar_all', 'activity', 'probe_activity', 'disc_act_acc', 1 / 5, 'unseen_dev_seen_act'),
    ('N1 HHAR all activities', 'validate_hhar_all', 'device', 'probe_device', 'disc_dev_acc', 1 / 2, 'seen_dev_unseen_act'),
    ('N2 HHAR bike held out', 'validate_hhar_bike', 'activity', 'probe_activity', 'disc_act_acc', 1 / 3, 'unseen_dev_seen_act'),
    ('N2 HHAR bike held out', 'validate_hhar_bike', 'device', 'probe_device', 'disc_dev_acc', 1 / 2, 'seen_dev_unseen_act'),
    ('N3 HHAR RMSprop', 'validate_hhar_rmsprop', 'activity', 'probe_activity', 'disc_act_acc', 1 / 3, 'unseen_dev_seen_act'),
    ('N3 HHAR RMSprop', 'validate_hhar_rmsprop', 'device', 'probe_device', 'disc_dev_acc', 1 / 2, 'seen_dev_unseen_act'),
]
FAM = (('', 'lin'), ('_mlp', 'mlp'), ('_knn', 'knn'))


def load(d):
    f = glob.glob(D + d + '/*.pkl')
    return pickle.load(open(f[0], 'rb')) if f else None


def lams(r):
    return sorted(r, key=lambda k: float(k.split('=')[1]))


def curves(r, stem, cell='seen_seen', suffix=''):
    ks = lams(r)
    return {n: np.array([r[k].get(f'{stem}{fs}_tc_{cell}{suffix}_raw', [np.nan]) for k in ks], float).T for fs, n in FAM}


def best(P):
    return np.nanmax(np.stack(list(P.values())), axis=0)


def rho(p0, p, c):
    return 100 * (p0 - p) / (p0 - c)


def analyse(cond):
    name, d, factor, stem, dk, c, ucell = cond
    r = load(d)
    if r is None:
        return None
    L = np.array([float(k.split('=')[1]) for k in lams(r)])
    full, A, B = curves(r, stem), curves(r, stem, suffix='_halfA'), curves(r, stem, suffix='_halfB')
    S, FA, FB = best(full), best(A), best(B)
    p0 = float(np.nanmean(S[:, 0]))
    head = p0 - c
    ns = S.shape[0]
    # naive best case on the whole cell (min over lambda per seed, lambda = 0 included)
    b1 = float(np.mean(rho(full['lin'][:, 0], np.nanmin(full['lin'], axis=1), c)))
    b2 = float(np.mean(rho(S[:, 0], np.nanmin(S, axis=1), c)))
    # protocol: per seed choose lambda on A (strongest), read on B
    ia = np.nanargmin(FA, axis=1)
    prot = float(np.mean([rho(FB[s, 0], FB[s, ia[s]], c) for s in range(ns)]))
    prot_lin = float(np.mean([rho(B['lin'][s, 0], B['lin'][s, np.nanargmin(A['lin'][s])], c) for s in range(ns)]))
    # support for a B1 -> P change: at the lambda B1 selected, strongest minus linear on half B, and on the unseen cell
    i1 = np.nanargmin(full['lin'], axis=1)
    d_B = np.array([FB[s, i1[s]] - B['lin'][s, i1[s]] for s in range(ns)])
    sup = dict(dB_mean=float(d_B.mean()), dB_min=float(d_B.min()))
    if ucell:
        U = curves(r, stem, cell=ucell)
        dU = np.array([best(U)[s, i1[s]] - U['lin'][s, i1[s]] for s in range(ns)])
        sup.update(dU_mean=float(np.nanmean(dU)), dU_min=float(np.nanmin(dU)))
    # use case
    dacc = np.array([r[k].get(dk, (np.nan, 0))[0] for k in lams(r)])
    ident = np.array([r[k]['cells']['seen_seen']['bout_acc'][0] for k in lams(r)])
    meanA, meanAl = np.nanmean(FA, axis=0), np.nanmean(A['lin'], axis=0)
    pick = {'fixed': int(np.argmin(abs(L - 1))), 'discriminator': int(np.nanargmin(dacc)), 'linear': int(np.nanargmin(meanAl)),
            'protocol': int(np.nanargmin(meanA))}
    flag = rho(np.nanmean(FA[:, 0]), np.nanmin(meanA), c) < R.RECOVERABLE_BELOW
    use = {m: dict(lam=float(L[i]), rho_B=float(np.mean([rho(FB[s, 0], FB[s, i], c) for s in range(ns)])),
                   rho_B_min=float(min(rho(FB[s, 0], FB[s, i], c) for s in range(ns))),
                   d_id=float(ident[i] - ident[0])) for m, i in pick.items()}
    use['protocol']['flag_no_benefit'] = bool(flag)
    use['recommend0'] = dict(lam=0.0, rho_B=0.0, rho_B_min=0.0, d_id=0.0)
    mind = float(np.nanmin(dacc))
    return dict(name=name, factor=factor, c=c, p0=p0, head=head, b1=b1, b2=b2, prot=prot, prot_lin=prot_lin,
                v1=R.verdict(b1, head), v2=R.verdict(b2, head), vp=R.verdict(prot, head), vlin_sel=R.verdict(prot_lin, head),
                sup=sup, use=use, disc_min=mind, ns=ns)


def show(rows, title):
    print(f'\n=== {title} ===')
    print(f"{'condition':<24}{'factor':<10}{'p0':>5}{'B1 lin':>8}{'B2 str':>8}{'P sel':>8}   {'B1 verdict':<16}{'B2 verdict':<16}{'P verdict':<16}"
          f"{'dB mean/min':>14}{'dU mean/min':>14}")
    for a in rows:
        s = a['sup']
        du = f"{s['dU_mean']:.3f}/{s['dU_min']:.3f}" if 'dU_mean' in s else '---'
        print(f"{a['name']:<24}{a['factor']:<10}{a['p0']:5.2f}{a['b1']:8.1f}{a['b2']:8.1f}{a['prot']:8.1f}   "
              f"{a['v1']:<16}{a['v2']:<16}{a['vp']:<16}{s['dB_mean']:7.3f}/{s['dB_min']:.3f}{du:>14}")
    n = len(rows)
    ch1 = sum(a['v1'] != a['vp'] for a in rows)
    ch2 = sum(a['v2'] != a['vp'] for a in rows)
    ch12 = sum(a['v1'] != a['v2'] for a in rows)
    print(f'{n} rows: B1 != P in {ch1}, B2 != P in {ch2}, B1 != B2 in {ch12}; no-headroom rows: {sum(a["vp"] == "no headroom" for a in rows)}')


def show_use(rows, title):
    print(f'\n=== {title}: choosing lambda (outcome on half B; rho in %, identity change in points) ===')
    print(f"{'condition':<24}{'factor':<10}" + ''.join(f'{m:>26}' for m in ('fixed lam=1', 'discriminator', 'linear', 'protocol')) + '  flag')
    for a in rows:
        u = a['use']
        cells = ''.join(f"  lam={u[m]['lam']:<5g} rho={u[m]['rho_B']:5.1f} id{100 * u[m]['d_id']:+5.1f}" for m in ('fixed', 'discriminator', 'linear', 'protocol'))
        print(f"{a['name']:<24}{a['factor']:<10}{cells}  {'NO BENEFIT' if u['protocol']['flag_no_benefit'] else ''}")
    for m in ('fixed', 'discriminator', 'linear', 'protocol'):
        print(f"  {m:<14} mean rho_B {np.mean([a['use'][m]['rho_B'] for a in rows]):6.1f}   mean identity change {100 * np.mean([a['use'][m]['d_id'] for a in rows]):+5.1f} pts   "
              f"worst rho_B {min(a['use'][m]['rho_B'] for a in rows):6.1f}")


if __name__ == '__main__':
    dev = [x for x in (analyse(c) for c in DEV) if x]
    new = [x for x in (analyse(c) for c in NEW) if x]
    show(dev, 'Development conditions')
    show_use(dev, 'Development conditions')
    if new:
        show(new, 'Validation conditions (rule frozen before the runs)')
        show_use(new, 'Validation conditions')
        print('\nPredictions P1-P4 (protocol_frozen.txt):')
        for a in new:
            print(f"  {a['name']:<24}{a['factor']:<9}headroom {a['head']:.2f}  verdict {a['vp']:<16} B1 {a['v1']:<16} min raw disc acc {a['disc_min']:.3f}")
