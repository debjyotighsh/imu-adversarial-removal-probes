"""
Export the numbers behind every data figure as CSV (for the journal's source-data request)
==========================================================================================
One file per figure, wide format: panel, series, x, n_seeds, mean, sd (sample SD, n-1), seed_1 ... seed_k.
The values are taken through the same loaders and the same arithmetic as the figure scripts, so they are
exactly what is drawn.

    python export_figure_data.py --out figure_data_regenerated
"""
import argparse
import csv
import glob
import os
import pickle

import numpy as np

import make_probe_figures as MPF
import make_seed_figures as MSF
import plot_dynamics as PD
import probe_gap_report as P
import seed_stats as SS
from probe_families_table import find_pkl

MAXSEED = 5


def write(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['panel', 'series', 'x', 'n_seeds', 'mean', 'sd'] + [f'seed_{i + 1}' for i in range(MAXSEED)])
        for panel, series, x, vals in rows:
            v = np.asarray(vals, float)
            ok = v[~np.isnan(v)]
            sd = float(np.std(ok, ddof=1)) if len(ok) > 1 else float('nan')
            w.writerow([panel, series, x, len(ok), f'{np.nanmean(v):.6f}', f'{sd:.6f}']
                       + [f'{a:.6f}' for a in v] + [''] * (MAXSEED - len(v)))
    print('wrote', path, len(rows), 'rows')


def lam_rows(panel, r, pk, dk):
    lam, probe, disc = MSF.curves(r, pk, dk)
    rows = []
    for j, l in enumerate(lam):
        rows.append((panel, 'post-hoc probe (held-out), activity', f'{l:g}', probe[:, j]))
        rows.append((panel, 'discriminator (final-epoch training accuracy), activity', f'{l:g}', disc[:, j]))
    return rows


def fig1_fig4(d, out):
    def load(frag, pred):
        # five-seed merged run where one exists (HHAR central, RealWorld full pool), as in the paper's figures
        return pickle.load(open(MPF.hits_for(d, frag, pred)[0], 'rb'))

    hh = load('windows_hz50_win2.56_ov0.5_bike', lambda b: '_cap' not in b and 'dlm5' in b and '_deg' not in b)
    rw = load('results_sweep_realworld', lambda b: '_cap' not in b)
    ms = load('results_sweep_motionsense', lambda b: '_cap' not in b and '_deg' not in b)
    mb = load('results_sweep_mobiact', lambda b: '_cap' not in b)
    pk, dk = 'probe_activity_tc_seen_seen', 'disc_act_acc'
    write(os.path.join(out, 'fig1_divergence.csv'), lam_rows('HHAR (ambulatory)', hh, pk, dk))
    rows = []
    for name, r in (('HHAR (ambulatory)', hh), ('RealWorld (full pool)', rw), ('MotionSense (ambulatory)', ms),
                    ('MobiAct (stairs up / down)', mb)):
        rows += lam_rows(name, r, pk, dk)
    write(os.path.join(out, 'fig4_lambda_band.csv'), rows)


def fig3(d, out):
    full, fac = MPF.load(d, 'RealWorld (ambulatory, full)')
    cap, _ = MPF.load(d, 'RealWorld (ambulatory, 15.3k)')
    stem = fac['activity'][0]
    rows = []
    for name, r, size in (('full', full, '102,898 windows'), ('cap', cap, '15,307 windows')):
        lam, Pd = SS.seed_curves(r, stem, 'seen_seen')
        best = np.nanmax(np.stack(list(Pd.values())), axis=0)
        for j, l in enumerate(lam):
            rows.append(('(a) linear probe', size, f'{l:g}', Pd['lin'][:, j]))
            rows.append(('(b) strongest probe (max of linear / MLP / k-NN)', size, f'{l:g}', best[:, j]))
    write(os.path.join(out, 'fig3_datasize.csv'), rows)


def dyn_rows(panel, series_name, x, arr):
    return [(panel, series_name, f'{xx:g}', arr[:, i]) for i, xx in enumerate(x)]


def fig5(pkl, out):
    D = pickle.load(open(sorted(glob.glob(pkl))[0], 'rb'))
    rows = []
    spec = [('(a) accuracy', 'discriminator, lambda={}', 'act_acc', ('0', '1')),
            ('(b) discriminator loss (cross-entropy)', 'lambda={}', 'act_loss', ('0', '0.1', '1')),
            ('(d) representation drift (relative)', 'lambda={}', 'rep_drift', ('0', '0.1', '1')),
            ('(e) discriminator-head gradient norm', 'lambda={}', 'gn_head_act', ('0', '0.1', '1'))]
    for panel, nm, field, lams in spec:
        for l in lams:
            if f'lambda={l}' in D:
                x, _, _ = PD.series(D[f'lambda={l}'], field)
                rows += dyn_rows(panel, nm.format(l), x, PD._LAST)
    x, _, _ = PD.series(D['lambda=1'], 'probe_act')
    rows += dyn_rows('(a) accuracy', 'post-hoc probe, lambda=1', x, PD._LAST)
    for nm, field in (('identity part', 'gn_enc_id'), ('adversarial part', 'gn_enc_adv_act')):
        x, _, _ = PD.series(D['lambda=1'], field)
        rows += dyn_rows('(c) encoder gradient norm, lambda=1', nm, x, PD._LAST)
    runs = D['lambda=1']
    for nm, which, field in (('trained head (fit split)', 'cont', 'train_acc'), ('trained head (held out)', 'cont', 'test_acc'),
                             ('fresh wide head (held out)', 'fresh', 'test_acc')):
        x, _, _ = PD.recovery_series(runs, which, field)
        rows += dyn_rows('(f) recovery on frozen encoder, lambda=1 (x = extra epochs)', nm, x, PD._LAST)
    write(os.path.join(out, 'fig5_dynamics.csv'), rows)


def fig7(pkl, out):
    D = pickle.load(open(sorted(glob.glob(pkl))[0], 'rb'))
    runs = D['lambda=1']
    x, _, _ = PD.series(runs, 'act_acc')
    rows = dyn_rows('(a) discriminator training accuracy (restart every 5 epochs)', 'discriminator (training)', x, PD._LAST)
    for nm, which in (('in-training head, continued', 'cont'), ('fresh wide head', 'fresh')):
        xr, _, _ = PD.recovery_series(runs, which, 'test_bal_acc')
        rows += dyn_rows('(b) recovery, balanced accuracy (x = extra epochs)', nm, xr, PD._LAST)
    write(os.path.join(out, 'fig7_restart.csv'), rows)


def probe_gap(out):
    def Gseed(r, lam, key):
        return (np.asarray(r[f'lambda={lam:g}'][key + '_raw'], float)
                - np.asarray(r[f'lambda={lam:g}']['disc_act_heldout_tc_seen_seen_raw'], float))

    runs = {'HHAR': P.load('hhar_ep25'), 'MobiAct B': P.load('mobiact_ep25')}
    rows = []
    groups = ['lr_C0.001', 'lr_C0.1', 'lr_C10', 'mlp_w4', 'mlp_w16', 'mlp_w64', 'mlp_w256', 'mlp_w1024', 'knn_k1', 'knn_k5', 'knn_k50']
    for name, r in runs.items():
        for g in groups:
            rows.append(('(a) probe capacity, lambda=1', name, g, Gseed(r, 1.0, f'probe_activity_tc_seen_seen_grid_{g}')))
        for fam in ('lr', 'mlp'):
            for n in (100, 300, 1000, 3000):
                rows.append(('(b) probe fit-set size, lambda=1', f'{name} {fam.upper()}', n,
                             Gseed(r, 1.0, f'probe_activity_tc_seen_seen_grid_fit{n}_{fam}')))

    def strongestG(name, lam):
        r = P.load(name)
        return P.strongest(r, lam) - np.asarray(r[f'lambda={lam:g}']['disc_act_heldout_tc_seen_seen_raw'], float)

    for lam in (1.0, 0.0):
        for n, nm in ((1000, 'hhar_n1000'), (3000, 'hhar_n3000'), (15315, 'hhar_ep25')):
            rows.append(('(c) encoder training windows (HHAR, 25 epochs)', f'lambda={lam:g}', n, strongestG(nm, lam)))
        for e in (5, 10, 25, 50, 100):
            rows.append(('(d) optimisation budget (HHAR)', f'lambda={lam:g}', e, strongestG(f'hhar_ep{e}', lam)))
    write(os.path.join(out, 'fig_probe_gap.csv'), rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--frozen', default='gait_experiment_output_v3_frozen')
    ap.add_argument('--dyn_pkl', default='gait_experiment_output_v2_dynamics/*_dynamics.pkl')
    ap.add_argument('--restart_pkl', default='gait_experiment_output_v3_restart_dynamics/*_dynamics.pkl')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig1_fig4(a.frozen, a.out)
    fig3(a.frozen, a.out)
    fig5(a.dyn_pkl, a.out)
    fig7(a.restart_pkl, a.out)
    probe_gap(a.out)


if __name__ == '__main__':
    main()
