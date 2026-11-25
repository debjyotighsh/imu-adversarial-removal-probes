"""IIDM on its native use case and on datasets with room to remove (output of run_iidm_native.ps1).

For each condition: held-out discriminator, three held-out probes and the strongest-probe removal rho_best
relative to the same baseline's lambda=0 reading (rho = (p0 - p_lambda) / (p0 - chance), strongest probe family
at each lambda; the rows give every lambda, the best case is the largest rho)."""
import glob
import pickle

import numpy as np

import frozen_analysis as FA


def load(d):
    f = glob.glob(d + '/*.pkl')
    return pickle.load(open(f[0], 'rb')) if f else None


def lam(k):
    return float(k.split('=')[1])


def pm(v, nd=3):
    v = np.asarray(v, float)
    return f"{np.nanmean(v):.{nd}f}±{(np.nanstd(v, ddof=1) if len(v) > 1 else 0):.{nd}f}"


def strongest(v, factor):
    """per-seed max over linear / MLP / k-NN held-out probe accuracy for the 'activity' or 'device' slot"""
    P = [np.asarray(v[f'probe_{factor}{s}_tc_seen_seen_raw'], float) for s in ('', '_mlp', '_knn')]
    return np.nanmax(np.stack(P), axis=0)


def table(title, res, base, factor, disc_key, chance, id_cell=None):
    print(f"\n=== {title} | factor slot '{factor}' | chance {chance:.3f} ===")
    p0 = float(np.mean(strongest(base['lambda=0'], factor))) if base and 'lambda=0' in base else float('nan')
    print(f"baseline strongest probe at lambda=0: {p0:.3f}")
    print(f"{'lambda':>7} {'disc held-out':>14} {'ent':>5} {'lin':>6} {'MLP':>6} {'kNN':>6} {'strongest':>9} {'rho_best%':>9}"
          + ("   id-task seen / other" if id_cell else ""))
    for k in sorted(res, key=lam):
        v = res[k]
        held = np.asarray(v[f'{disc_key}_heldout_tc_seen_seen_raw'], float)
        confs = v.get('disc_act_conf_raw') if factor == 'activity' else v.get('disc_dev_conf_raw')
        ent = np.mean([FA.conf_stats(c)[2] for c in confs]) if confs is not None else float('nan')
        lin, mlp, knn = (np.mean(v[f'probe_{factor}{s}_tc_seen_seen_raw']) for s in ('', '_mlp', '_knn'))
        st = float(np.mean(strongest(v, factor)))
        rho = 100 * (p0 - st) / (p0 - chance)
        extra = ''
        if id_cell:
            c = v['cells']
            extra = f"   {c['seen_seen']['window_acc'][0]:.3f} / {c[id_cell]['window_acc'][0]:.3f}"
        print(f"{lam(k):7g} {pm(held):>14} {ent:5.2f} {lin:6.3f} {mlp:6.3f} {knn:6.3f} {st:9.3f} {rho:9.1f}{extra}")


# --- MotionSense, subject identity as the nuisance (IIDM's native use case) -----------------------
ms_base = load('gait_experiment_output_v3_subject_nuisance_discheldout')
for name in ('ms_simplified', 'ms_faithful'):
    r = load(f'gait_experiment_output_v3_iidmn_{name}')
    if r:
        table(f'MotionSense swap, IIDM {name}', r, ms_base, 'activity', 'disc_act', 1 / 19, id_cell='unseen_subj')
if ms_base:
    table('MotionSense swap, GRL baseline', ms_base, ms_base, 'activity', 'disc_act', 1 / 19, id_cell='unseen_subj')

# --- HHAR and RealWorld ------------------------------------------------------------------------------
hh_base = load('gait_experiment_output_v3_disc_heldout')
for name in ('hhar_faithful', 'hhar_long'):
    r = load(f'gait_experiment_output_v3_iidmn_{name}')
    if r:
        table(f'HHAR {name}', r, hh_base, 'activity', 'disc_act', 1 / 3, id_cell='seen_dev_unseen_act')
        table(f'HHAR {name}', r, hh_base, 'device', 'disc_dev', 1 / 2)
rw_base = load('gait_experiment_output_v3_restart2_realworld_base')
r = load('gait_experiment_output_v3_iidmn_rw_faithful')
if r:
    table('RealWorld (capped) rw_faithful', r, rw_base, 'activity', 'disc_act', 1 / 3, id_cell='seen_dev_unseen_act')
    table('RealWorld (capped) rw_faithful', r, rw_base, 'device', 'disc_dev', 1 / 6)
if rw_base:
    table('RealWorld (capped) GRL baseline', rw_base, rw_base, 'activity', 'disc_act', 1 / 3, id_cell='seen_dev_unseen_act')
    table('RealWorld (capped) GRL baseline', rw_base, rw_base, 'device', 'disc_dev', 1 / 6)
