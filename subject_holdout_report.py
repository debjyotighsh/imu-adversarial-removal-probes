"""Held-out subject evaluation (output of run_subject_holdout.ps1).

For every condition and factor, three runs hold out three different random sets of whole subjects (splits 1-3, one encoder seed each).
Held-out subjects are absent from encoder training, the probe fit pool and the 'seen_seen' reference cell. For lambda = 0 and each lambda
run we report, mean over splits (range in brackets):
  S_adv      balanced accuracy of the discriminator head on the cell
  S_probe    strongest of the linear, MLP and k-NN probes on the cell
  G          S_probe - S_adv, with the mean over splits of the segment-bootstrap 95% interval
for 'seen_seen' (windows of seen subjects the encoder did not train on) and 'unseen_subject' (windows of the held-out subjects).
Usage: python subject_holdout_report.py [--latex]
"""
import glob
import pickle
import sys

import numpy as np

D = 'gait_experiment_output_v3_subj_'
CONDS = [  # label, dir stem, factor, probe key, disc key, chance, lambdas
    ('HHAR', 'hhar', 'activity', 'probe_activity', 'disc_act', 1 / 3, (0.0, 1.0)),
    ('HHAR', 'hhar', 'device', 'probe_device', 'disc_dev', 1 / 2, (0.0, 1.0)),
    ('MobiAct B', 'mobiact_B', 'activity', 'probe_activity', 'disc_act', 1 / 2, (0.0, 1.0)),
    ('MotionSense', 'motionsense_amb', 'activity', 'probe_activity', 'disc_act', 1 / 3, (0.0, 0.15, 1.0)),
    ('RealWorld', 'realworld_cap', 'activity', 'probe_activity', 'disc_act', 1 / 3, (0.0, 0.15, 1.0)),
    ('RealWorld', 'realworld_cap', 'position', 'probe_device', 'disc_dev', 1 / 6, (0.0, 0.15, 1.0)),
]


def load(stem):
    out = []
    for s in (1, 2, 3):
        f = glob.glob(f'{D}{stem}_s{s}/*.pkl')
        if f:
            out.append(pickle.load(open(f[0], 'rb')))
    return out


def val(r, lam, key):
    k = f'lambda={lam:g}'
    if k not in r:
        return np.nan
    v = r[k].get(key + '_raw', [np.nan])
    return float(np.nanmean(v))


def strongest(r, lam, pk, cell):
    return np.nanmax([val(r, lam, f'{pk}{s}_tc_{cell}') for s in ('', '_mlp', '_knn')])


def stats(rs, fn):
    v = np.array([fn(r) for r in rs], float)
    return np.nanmean(v), np.nanmin(v), np.nanmax(v), v


def collect():
    rows = []
    for label, stem, factor, pk, dk, c, lams in CONDS:
        rs = load(stem)
        if not rs:
            continue
        for lam in lams:
            row = dict(label=label, factor=factor, lam=lam, c=c, n=len(rs))
            for cell in ('seen_seen', 'unseen_subject'):
                row[cell] = dict(
                    adv=stats(rs, lambda r: val(r, lam, f'{dk}_heldout_tc_{cell}')),
                    probe=stats(rs, lambda r: strongest(r, lam, pk, cell)),
                    G=stats(rs, lambda r: strongest(r, lam, pk, cell) - val(r, lam, f'{dk}_heldout_tc_{cell}')),
                    lo=np.nanmean([val(r, lam, f'{pk}_tc_{cell}_boot_G_lo') for r in rs]),
                    hi=np.nanmean([val(r, lam, f'{pk}_tc_{cell}_boot_G_hi') for r in rs]),
                    p0=stats(rs, lambda r: strongest(r, 0.0, pk, cell))[0])
            rows.append(row)
    return rows


def f2(x):
    return f'{0.0 if round(x, 2) == 0 else x:.2f}'


if __name__ == '__main__':
    rows = collect()
    if '--latex' in sys.argv:
        prev = None
        for r in rows:
            head = f"{r['label']}, {r['factor']}"
            if prev is not None and head != prev:
                print(r'\addlinespace')
            first = head if head != prev else ''
            prev = head
            s, u = r['seen_seen'], r['unseen_subject']
            print(f"{first} & {r['lam']:g} & ${f2(s['adv'][0])}$ & ${f2(s['probe'][0])}$ & ${f2(s['G'][0])}$ & "
                  f"${f2(u['adv'][0])}$ & ${f2(u['probe'][0])}$ & ${f2(u['G'][0])}$ & [${f2(u['lo'])}$, ${f2(u['hi'])}$] & "
                  f"${f2(u['G'][1])}$--${f2(u['G'][2])}$ \\\\")
        sys.exit()
    print(f"{'condition':<13}{'factor':<10}{'lam':>5}{'n':>3} | seen subjects: S_adv S_probe    G | unseen subjects: S_adv S_probe    G   [95% CI]   range over splits")
    for r in rows:
        s, u = r['seen_seen'], r['unseen_subject']
        print(f"{r['label']:<13}{r['factor']:<10}{r['lam']:5g}{r['n']:3d} | {s['adv'][0]:20.3f}{s['probe'][0]:7.3f}{s['G'][0]:6.2f} | "
              f"{u['adv'][0]:22.3f}{u['probe'][0]:7.3f}{u['G'][0]:6.2f}  [{u['lo']:.2f},{u['hi']:.2f}]  {u['G'][1]:.2f}-{u['G'][2]:.2f}")
