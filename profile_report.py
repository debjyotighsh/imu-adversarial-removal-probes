"""Recoverability profile (paper Section "The Probe-Generalisation Gap" and Table "Recoverability profile").

For each condition, factor and lambda, from the --probe_grid runs of run_profile.ps1 (three seeds, mean; ranges are over seeds):
  c            chance of the balanced accuracy (uniform over the classes; a constant predictor also scores c on this metric)
  p0           strongest probe at lambda = 0 (headroom is p0 - c)
  S_adv        balanced accuracy of the training discriminator head on the held-out cell
  S_probe      strongest of the linear, MLP and k-NN probes on the same cell, with the 95% interval over test segments
  G            S_probe - S_adv, with the segment-bootstrap interval (mean of the three seeds' interval ends)
  G_grid       smallest and largest G over the probe grid (LR C, MLP width/depth, k-NN k; every probe fitted on >= 1,000 windows)
  d_opt        most extra recovery the attacker obtains from budget: max over MLP epochs {5, 20, 100} and five restarts, minus the default MLP
  d_n          recovery gained from 300 to 3,000 fit windows (MLP)
  G_gen        G on the generalisation cell (unseen device for activity; unseen activity for device / position)
Usage: python profile_report.py [--latex]
"""
import glob
import pickle
import sys

import numpy as np

D = 'gait_experiment_output_v3_profile_'
# (label, directory suffix, factor label, probe key, disc key, chance, generalisation cell, lambdas)
CONDS = [
    ('HHAR', 'hhar', 'activity', 'probe_activity', 'disc_act', 1 / 3, 'unseen_dev_seen_act', (0.0, 1.0)),
    ('HHAR', 'hhar', 'device', 'probe_device', 'disc_dev', 1 / 2, 'seen_dev_unseen_act', (0.0, 1.0)),
    ('MobiAct B', 'mobiact_B', 'activity', 'probe_activity', 'disc_act', 1 / 2, None, (0.0, 1.0)),
    ('MotionSense', 'motionsense_amb', 'activity', 'probe_activity', 'disc_act', 1 / 3, None, (0.0, 0.15, 1.0)),
    ('RealWorld', 'realworld_cap', 'activity', 'probe_activity', 'disc_act', 1 / 3, 'unseen_dev_seen_act', (0.0, 0.15, 1.0)),
    ('RealWorld', 'realworld_cap', 'position', 'probe_device', 'disc_dev', 1 / 6, 'seen_dev_unseen_act', (0.0, 0.15, 1.0)),
]


def raw(r, lam, key):
    k = f'lambda={lam:g}'
    return np.asarray(r[k].get(key + '_raw', [np.nan]), float) if k in r else np.array([np.nan])


def best3(r, lam, pk, cell='seen_seen'):
    return np.nanmax(np.stack([raw(r, lam, f'{pk}{s}_tc_{cell}') for s in ('', '_mlp', '_knn')]), axis=0)


def grid(r, lam, pk, name):
    return raw(r, lam, f'{pk}_tc_seen_seen_grid_{name}')


GRID_BIG = ['lr_C0.001', 'lr_C0.1', 'lr_C10', 'mlp_w4', 'mlp_w16', 'mlp_w64', 'mlp_w256', 'mlp_w1024', 'mlp_256x256',
            'knn_k1', 'knn_k5', 'knn_k50', 'fit1000_lr', 'fit1000_mlp', 'fit3000_lr', 'fit3000_mlp']


def profile(r, lam, pk, dk, c, gen_cell, p0):
    disc = raw(r, lam, f'{dk}_heldout_tc_seen_seen')
    S = best3(r, lam, pk)
    G = S - disc
    gG = np.stack([grid(r, lam, pk, n) for n in GRID_BIG]) - disc      # probes x seeds
    mlp = raw(r, lam, f'{pk}_mlp_tc_seen_seen')
    opt = np.nanmax(np.stack([grid(r, lam, pk, f'mlpbudget_e{e}') for e in (5, 20, 100)] + [grid(r, lam, pk, 'mlprestart_max')]), axis=0)
    row = dict(
        c=c, p0=p0, adv=np.nanmean(disc), adv_rng=(np.nanmin(disc), np.nanmax(disc)), S=np.nanmean(S),
        S_ci=(np.nanmean(raw(r, lam, f'{pk}_tc_seen_seen_boot_probe_lo')), np.nanmean(raw(r, lam, f'{pk}_tc_seen_seen_boot_probe_hi'))),
        G=np.nanmean(G), G_rng=(np.nanmin(G), np.nanmax(G)),
        G_ci=(np.nanmean(raw(r, lam, f'{pk}_tc_seen_seen_boot_G_lo')), np.nanmean(raw(r, lam, f'{pk}_tc_seen_seen_boot_G_hi'))),
        G_grid=(np.nanmean(gG, axis=1).min(), np.nanmean(gG, axis=1).max()),
        lin=np.nanmean(raw(r, lam, f'{pk}_tc_seen_seen')), mlp=np.nanmean(mlp), knn=np.nanmean(raw(r, lam, f'{pk}_knn_tc_seen_seen')),
        d_opt=np.nanmean(opt - mlp),
        d_n=np.nanmean(grid(r, lam, pk, 'fit3000_mlp') - grid(r, lam, pk, 'fit300_mlp')),
        major=np.nanmean(raw(r, lam, f'{pk}_tc_seen_seen_majority_raw')),
    )
    if gen_cell:
        dg = raw(r, lam, f'{dk}_heldout_tc_{gen_cell}')
        sg = best3(r, lam, pk, gen_cell)
        row['G_gen'] = np.nanmean(sg - dg)
        row['G_gen_ci'] = (np.nanmean(raw(r, lam, f'{pk}_tc_{gen_cell}_boot_G_lo')), np.nanmean(raw(r, lam, f'{pk}_tc_{gen_cell}_boot_G_hi')))
        row['S_gen'] = np.nanmean(sg)
    return row


def collect():
    rows = []
    for label, suf, factor, pk, dk, c, gen, lams in CONDS:
        f = glob.glob(D + suf + '/*.pkl')
        if not f:
            continue
        r = pickle.load(open(f[0], 'rb'))
        p0 = float(np.nanmean(best3(r, 0.0, pk)))
        for lam in lams:
            if f'lambda={lam:g}' not in r:
                continue
            rows.append((label, factor, lam, profile(r, lam, pk, dk, c, gen, p0)))
    return rows


def fmt(x, d=2):
    return '---' if x is None or not np.isfinite(x) else f'{x:.{d}f}'


def z(x, d=2):
    v = round(float(x), d)
    return f'{0.0 if v == 0 else v:.{d}f}'   # no negative zero


def latex(rows):
    out = []
    prev = None
    for label, factor, lam, p in rows:
        head = f'{label}, {factor}'
        if prev is not None and head != prev:
            out.append(r'\addlinespace')
        first = head if head != prev else ''
        prev = head
        gen = f"${z(p['G_gen'])}$" if 'G_gen' in p else '---'
        out.append(f"{first} & {lam:g} & ${p['c']:.2f}$ & ${p['p0']:.2f}$ & ${z(p['adv'])}$ & "
                   f"${z(p['S'])}$ & ${z(p['G'])}$ & [${z(p['G_ci'][0])}$, ${z(p['G_ci'][1])}$] & "
                   f"${z(p['G_grid'][0])}$ to ${z(p['G_grid'][1])}$ & ${z(p['d_opt'], 3)}$ & ${z(p['d_n'])}$ & {gen} \\\\")
    print('\n'.join(out))


if __name__ == '__main__':
    rows = collect()
    if '--latex' in sys.argv:
        latex(rows)
        sys.exit()
    print(f"{'condition':<12}{'factor':<10}{'lam':>5}{'c':>6}{'p0':>6}{'S_adv':>7}{'S_probe [95% CI]':>22}{'G [95% CI]':>22}"
          f"{'G_grid':>14}{'d_opt':>7}{'d_n':>7}{'G_gen [CI]':>20}")
    for label, factor, lam, p in rows:
        gen = f"{fmt(p['G_gen'])} [{fmt(p['G_gen_ci'][0])},{fmt(p['G_gen_ci'][1])}]" if 'G_gen' in p else '---'
        print(f"{label:<12}{factor:<10}{lam:5g}{p['c']:6.2f}{p['p0']:6.2f}{p['adv']:7.2f}"
              f"{fmt(p['S']):>8} [{fmt(p['S_ci'][0])},{fmt(p['S_ci'][1])}]"
              f"{fmt(p['G']):>8} [{fmt(p['G_ci'][0])},{fmt(p['G_ci'][1])}]"
              f"{fmt(p['G_grid'][0]):>8}-{fmt(p['G_grid'][1])}{fmt(p['d_opt'], 3):>8}{fmt(p['d_n'], 3):>8}{gen:>20}")
