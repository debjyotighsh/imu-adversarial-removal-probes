"""Verdict table of the multi-axis protocol (paper Section "A Multi-Axis Evaluation Protocol", Table of verdicts).

Reads the split-half runs of run_selection.ps1. For each condition and factor: p0 (linear, whole cell, lambda=0), the selected
reduction rho (lambda chosen per seed on half A, read on half B; mean over seeds) for the linear and the strongest of three probes,
and the verdicts under the decision rule: no headroom if p0 - c < 0.05; recoverable if rho < 10 %; partly reduced if 10 % <= rho < 90 %;
not decoded by the tested probes if rho >= 90 %. The thresholds are conventions."""
import glob
import pickle

import numpy as np

D = 'gait_experiment_output_v3_'
CONDS = [
    ('HHAR', D + 'select_hhar', 'activity', 'probe_activity', 1 / 3),
    ('HHAR', D + 'select_hhar', 'device', 'probe_device', 1 / 2),
    ('RealWorld, 102.9k', D + 'select_realworld_full', 'activity', 'probe_activity', 1 / 3),
    ('RealWorld, 102.9k', D + 'select_realworld_full', 'position', 'probe_device', 1 / 6),
    ('RealWorld, 15.3k', D + 'select_realworld_cap', 'activity', 'probe_activity', 1 / 3),
    ('RealWorld, 15.3k', D + 'select_realworld_cap', 'position', 'probe_device', 1 / 6),
    ('MobiAct B (stairs)', D + 'select_mobiact_B', 'activity', 'probe_activity', 1 / 2),
    ('MobiAct A (jog/stand)', D + 'select_mobiact_A', 'activity', 'probe_activity', 1 / 2),
    ('MotionSense, ambulatory', D + 'select_motionsense_amb', 'activity', 'probe_activity', 1 / 3),
    ('MotionSense, all', D + 'select_motionsense_all', 'activity', 'probe_activity', 1 / 5),
]


def verdict(rho, headroom):
    if headroom < 0.05:
        return 'no headroom'
    return 'recoverable' if rho < 10 else ('partly reduced' if rho < 90 else 'not decoded by the tested probes')


def curves(r, stem, suffix=''):
    ks = sorted(r, key=lambda k: float(k.split('=')[1]))
    return {n: np.array([r[k].get(f'{stem}{fs}_tc_seen_seen{suffix}_raw', [np.nan]) for k in ks], float).T
            for fs, n in (('', 'lin'), ('_mlp', 'mlp'), ('_knn', 'knn'))}


def sel_rho(A, B, c):
    ist = np.nanargmin(A, axis=1)
    return 100 * np.array([(B[s, 0] - B[s, ist[s]]) / (B[s, 0] - c) for s in range(A.shape[0])])


print(f"{'condition':<26}{'factor':<10}{'p0':>7}{'rho_lin':>9}{'rho_str':>9}  {'linear-only verdict':<18} protocol verdict")
for name, d, factor, stem, c in CONDS:
    r = pickle.load(open(glob.glob(d + '/*.pkl')[0], 'rb'))
    full, A, B = curves(r, stem), curves(r, stem, '_halfA'), curves(r, stem, '_halfB')
    p0 = float(np.nanmean(full['lin'][:, 0]))
    best = lambda P: np.nanmax(np.stack(list(P.values())), axis=0)
    rl = float(np.mean(sel_rho(A['lin'], B['lin'], c)))
    rs = float(np.mean(sel_rho(best(A), best(B), c)))
    print(f"{name:<26}{factor:<10}{p0:7.3f}{rl:9.1f}{rs:9.1f}  {verdict(rl, p0 - c):<18} {verdict(rs, p0 - c)}")
