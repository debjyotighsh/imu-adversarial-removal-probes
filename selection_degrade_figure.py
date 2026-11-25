"""fig6_controlled.pdf and the table rows for the controlled degradation, with lambda chosen on separate data.

Reads the split-half reruns (run_selection_degrade.ps1 + run_selection_degrade_fill.ps1) and the matching baselines
(run_selection.ps1, first three seeds). Plots the SELECTED reduction (lambda chosen per seed on half A of the cell, rho read on
half B) for the linear probe and the strongest of three families against whole-cell prior decodability, and prints LaTeX rows
with both the best case and the selected value."""
import glob
import os
import pickle
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import make_figures as MF

D = 'gait_experiment_output_v3_'
F = np.sqrt(1.5)           # population SD of 3 -> sample SD


def load(path):
    return pickle.load(open(path, 'rb'))


def curves(r, lams, suffix=''):
    keys = [f'lambda={l:g}' for l in lams]
    out = {}
    for fs, name in (('', 'lin'), ('_mlp', 'mlp'), ('_knn', 'knn')):
        rows = [r[k].get(f'probe_activity{fs}_tc_seen_seen{suffix}_raw', [np.nan] * 3)[:3] for k in keys]
        out[name] = np.array(rows, float).T
    return out


def est(r, lams, c):
    full, A, B = curves(r, lams), curves(r, lams, '_halfA'), curves(r, lams, '_halfB')
    out = {'p0': full['lin'][:, 0], 'drop': full['lin'][:, 0] - np.nanmin(full['lin'], axis=1)}
    for fam, f in (('lin', lambda P: P['lin']), ('best', lambda P: np.nanmax(np.stack(list(P.values())), axis=0))):
        Fu, Aa, Bb = f(full), f(A), f(B)
        n = Fu.shape[0]
        out[fam + '_best'] = np.array([(Fu[s, 0] - np.nanmin(Fu[s])) / (Fu[s, 0] - c) for s in range(n)])
        ist = np.nanargmin(Aa, axis=1)
        out[fam + '_sel'] = np.array([(Bb[s, 0] - Bb[s, ist[s]]) / (Bb[s, 0] - c) for s in range(n)])
    cell = r['lambda=0']['cells']['seen_seen']['bout_acc']
    out['ident'] = (cell[0], cell[1] * F)
    return out


def rows_for(base_dir, glob_pat, lams, c):
    out = []
    b = glob.glob(base_dir + '/*.pkl')
    out.append(('none', 0.0, est(load(b[0]), lams, c)))
    for p in sorted(glob.glob(glob_pat)):
        m = re.search(r'_deg(noise|mix)([0-9.]+)\.pkl$', p)
        if m and 'segdetail' not in p and 'dynamics' not in p:
            out.append((m.group(1), float(m.group(2)), est(load(p), lams, c)))
    out.sort(key=lambda x: ({'none': 0, 'noise': 1, 'mix': 2}[x[0]], x[1]))
    return out


c = 1 / 3
hh = rows_for(D + 'select_hhar', D + 'select_degrade/*windows_hz50*deg*.pkl', [0, 0.05, 0.1, 0.25, 1], c)
ms = rows_for(D + 'select_motionsense_amb', D + 'select_degrade/*motionsense*deg*.pkl', [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1], c)


def pm(x, nd=1, scale=100.0):
    x = np.asarray(x, float) * scale
    return f"${np.nanmean(x):.{nd}f}\\pm{np.nanstd(x, ddof=1):.{nd}f}$"


for name, rows in (('HHAR', hh), ('MotionSense', ms)):
    print(f'%% {name}')
    for mode, lev, e in rows:
        if float(np.mean(e['p0'])) - c < 0.05:
            print(f"%% {mode} {lev:g}: p0 {np.mean(e['p0']):.3f} within 0.05 of chance, rho undefined")
            continue
        lab = 'none & ---' if mode == 'none' else f'{mode} & {lev:g}'
        print(f"{lab} & {pm(e['p0'], 3, 1)} & {pm(e['lin_best'])} & {pm(e['lin_sel'])} & {pm(e['best_best'])} & {pm(e['best_sel'])} "
              f"& {pm(e['drop'], 3, 1)} \\\\")

MF.style()
fig, axes = plt.subplots(1, 2, figsize=(MF.DBL_W, 2.6), sharey=True)
mk = {'none': 's', 'noise': 'o', 'mix': '^'}
for ax, (name, rows) in zip(axes, (('HHAR (ambulatory)', hh), ('MotionSense (ambulatory)', ms))):
    for mode, lev, e in rows:
        x = float(np.mean(e['p0']))
        if x - c < 0.05:
            continue
        for key, color, off in (('lin_sel', MF.C_BLUE, -0.004), ('best_sel', MF.C_ORANGE, 0.004)):
            v = 100 * e[key]
            ax.scatter(np.full(len(v), x + off), v, s=6, color=color, alpha=0.35, linewidths=0, zorder=2)
            ax.errorbar(x + off, np.nanmean(v), yerr=np.nanstd(v, ddof=1), fmt=mk[mode], color=color,
                        ms=4.4, elinewidth=0.7, capsize=1.5, mec=color, mfc=(color if key == 'lin_sel' else 'white'), mew=0.9, zorder=3)
    ax.axhline(0, color=MF.INK_MUTED, lw=0.5, ls=(0, (1, 2)), zorder=1)
    ax.set_title(name, loc='left', fontsize=7.5, pad=3)
    ax.invert_xaxis()
    MF.tidy(ax, xlabel=r'prior decodability (held-out linear probe, $\lambda=0$)')
axes[0].set_ylabel(r'selected reduction, $\rho$ (%)')
h = [plt.Line2D([], [], marker='s', color=MF.C_BLUE, ls='', ms=4, label=r'linear (filled)'),
     plt.Line2D([], [], marker='s', color=MF.C_ORANGE, ls='', ms=4, mfc='white', mew=0.9, label=r'strongest (open)'),
     plt.Line2D([], [], marker='s', color=MF.INK_MUTED, ls='', ms=4, label='baseline'),
     plt.Line2D([], [], marker='o', color=MF.INK_MUTED, ls='', ms=4, label='noise'),
     plt.Line2D([], [], marker='^', color=MF.INK_MUTED, ls='', ms=4, label='class mix')]
axes[0].legend(handles=h, frameon=False, loc='upper left', fontsize=6, handletextpad=0.2)
fig.tight_layout(pad=0.4, w_pad=1.0)
for o in ('./figures',):
    os.makedirs(o, exist_ok=True)
    fig.savefig(os.path.join(o, 'fig6_controlled.pdf'))
print('wrote fig6_controlled.pdf (selected reductions)')
