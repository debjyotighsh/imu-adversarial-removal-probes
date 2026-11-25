"""Probe-generalisation gap (paper Section "The Probe-Generalisation Gap").

Definition. For a trained encoder with an adversarial head and a factor with c = chance,
    G = A_probe - A_disc,
where A_disc is the balanced accuracy of the training-time discriminator head and A_probe the balanced accuracy of a probe fitted
post hoc to the frozen embedding, both scored on the SAME held-out evaluation cell. Fitted probes give a lower bound on what is
decodable, so G > 0 is recoverable information the discriminator does not report. We also give the normalised gap
    g = G / (p0 - c),   p0 = probe accuracy of the lambda = 0 encoder, so the scale is the headroom,
and report which axis moves it:
    E1 probe capacity (LR C, MLP width/depth, k-NN k) and probe fit-set size   [gait_experiment_output_v3_gap_hhar_ep25, _mobiact_ep25]
    E2 encoder training-set size                                                [gap_hhar_n1000, _n3000, hhar_ep25 = full]
    E3 optimisation budget (epochs)                                             [gap_hhar_ep5, _ep10, _ep25, _ep50, _ep100]
    E4 evaluation split: the discriminator on its own training pool vs held-out seen/seen cell, and the probe on seen/seen vs the
       unseen-device cell (both from the same runs).
Usage: python probe_gap_report.py [--figure out.png]
"""
import glob
import pickle
import sys

import numpy as np

D = 'gait_experiment_output_v3_gap_'
C_HHAR, C_MOB = 1 / 3, 1 / 2


def load(name):
    f = glob.glob(D + name + '/*.pkl')
    return pickle.load(open(f[0], 'rb')) if f else None


def vals(r, lam, key):
    k = f'lambda={lam:g}'
    if k not in r:
        return np.array([np.nan])
    return np.asarray(r[k].get(key + '_raw', [np.nan]), float)


def ms(x, d=3):
    x = np.asarray(x, float)
    return f'{np.nanmean(x):.{d}f}' + (f'$\\pm${np.nanstd(x):.{d}f}' if np.isfinite(x).sum() > 1 else '')


def strongest(r, lam, cell='seen_seen', stem='probe_activity'):
    p = np.stack([vals(r, lam, f'{stem}{s}_tc_{cell}') for s in ('', '_mlp', '_knn')])
    return np.nanmax(p, axis=0)


def gap_row(r, lam, c, cell='seen_seen'):
    disc = vals(r, lam, f'disc_act_heldout_tc_{cell}')
    pro = strongest(r, lam, cell)
    p0 = np.nanmean(strongest(r, 0.0, cell))
    return disc, pro, pro - disc, (pro - disc) / (p0 - c)


def main_table(name, c, lams):
    r = load(name)
    if r is None:
        print(f'[{name}: not available]'); return
    print(f'\n== {name}  (activity, seen/seen cell; chance {c:.3f}) ==')
    print(f"{'lam':>6}{'disc_heldout':>14}{'disc_train':>12}{'lin':>14}{'mlp':>14}{'knn':>14}{'strongest':>14}{'G':>14}{'g':>8}")
    for lam in lams:
        disc, pro, G, g = gap_row(r, lam, c)
        print(f"{lam:6g}{ms(disc):>14}{ms(vals(r, lam, 'disc_act_acc')):>12}"
              f"{ms(vals(r, lam, 'probe_activity_tc_seen_seen')):>14}{ms(vals(r, lam, 'probe_activity_mlp_tc_seen_seen')):>14}"
              f"{ms(vals(r, lam, 'probe_activity_knn_tc_seen_seen')):>14}{ms(pro):>14}{ms(G):>14}{np.nanmean(g):8.2f}")


def grid_table(name, c, lams):
    r = load(name)
    if r is None:
        return
    names = sorted({k.split('_grid_')[1][:-4] for lam in lams if f'lambda={lam:g}' in r for k in r[f'lambda={lam:g}']
                    if '_grid_' in k and k.endswith('_raw')})
    print(f'\n== {name}: probe grid, balanced accuracy on seen/seen (gap to held-out discriminator in brackets) ==')
    print(f"{'probe':<14}" + ''.join(f'{"lam=" + format(l, "g"):>22}' for l in lams))
    disc = {l: np.nanmean(vals(r, l, 'disc_act_heldout_tc_seen_seen')) for l in lams}
    for n in names:
        row = ''
        for l in lams:
            v = np.nanmean(vals(r, l, f'probe_activity_tc_seen_seen_grid_{n}'))
            row += f'{v:12.3f} ({v - disc[l]:+.3f})'
        print(f'{n:<14}{row}')


def axis_table(label, names, c, lams=(0.0, 1.0)):
    print(f'\n== {label} ==')
    print(f"{'setting':<14}" + ''.join(f'{"disc(lam=" + format(l, "g") + ")":>13}{"probe":>8}{"G":>9}' for l in lams))
    for tag, name in names:
        r = load(name)
        if r is None:
            print(f'{tag:<14} [missing]'); continue
        row = ''
        for l in lams:
            disc, pro, G, g = gap_row(r, l, c)
            row += f'{np.nanmean(disc):13.3f}{np.nanmean(pro):8.3f}{np.nanmean(G):9.3f}'
        print(f'{tag:<14}{row}')


def split_table(name, c, lams):
    r = load(name)
    if r is None:
        return
    print(f'\n== E4 evaluation split ({name}) ==')
    print(f"{'lam':>6}{'disc train':>12}{'disc seen/seen':>16}{'disc unseen-dev':>17}{'probe seen/seen':>17}{'probe unseen-dev':>18}"
          f"{'G seen':>9}{'G unseen':>10}")
    for lam in lams:
        ds = vals(r, lam, 'disc_act_heldout_tc_seen_seen')
        du = vals(r, lam, 'disc_act_heldout_tc_unseen_dev_seen_act')
        ps = strongest(r, lam, 'seen_seen')
        pu = strongest(r, lam, 'unseen_dev_seen_act')
        print(f"{lam:6g}{np.nanmean(vals(r, lam, 'disc_act_acc')):12.3f}{np.nanmean(ds):16.3f}{np.nanmean(du):17.3f}"
              f"{np.nanmean(ps):17.3f}{np.nanmean(pu):18.3f}{np.nanmean(ps - ds):9.3f}{np.nanmean(pu - du):10.3f}")


if __name__ == '__main__':
    L = (0.0, 0.1, 0.25, 1.0)
    main_table('hhar_ep25', C_HHAR, L)
    main_table('mobiact_ep25', C_MOB, (0.0, 0.5, 1.0))
    grid_table('hhar_ep25', C_HHAR, L)
    grid_table('mobiact_ep25', C_MOB, (0.0, 0.5, 1.0))
    axis_table('E2 encoder training-set size (HHAR)', [('1000', 'hhar_n1000'), ('3000', 'hhar_n3000'), ('full', 'hhar_ep25')], C_HHAR)
    axis_table('E3 optimisation budget, epochs (HHAR)', [(str(e), f'hhar_ep{e}') for e in (5, 10, 25, 50, 100)], C_HHAR)
    split_table('hhar_ep25', C_HHAR, L)
    split_table('mobiact_ep25', C_MOB, (0.0, 0.5, 1.0))
