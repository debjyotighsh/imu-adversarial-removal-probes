"""fig6_controlled.pdf from the frozen rerun, held-out probe (probe_activity_tc_seen_seen)."""
import glob
import os
import pickle
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import make_figures as MF

D = 'gait_experiment_output_v3_frozen'


def load_grid(r, stem, tc, lams):
    fam = {}
    for suf, name in (('', 'lin'), ('_mlp', 'mlp'), ('_knn', 'knn')):
        rows = []
        for l in lams:
            raw = r[f'lambda={l:g}'].get(f'{stem}{suf}_tc_{tc}_raw')
            rows.append(raw if raw is not None else [np.nan] * 3)
        fam[name] = np.array(rows, float).T
    return fam


def rho_seed(p, c):
    p0 = p[:, 0]
    with np.errstate(invalid='ignore', divide='ignore'):
        return (p0 - np.nanmin(p, axis=1)) / (p0 - c)


def dataset_rows(frag_pred, lams, c):
    b0 = pickle.load(open(
        [p for p in glob.glob(os.path.join(D, '*.pkl')) if frag_pred(p)
         and '_deg' not in p and 'segdetail' not in p and 'dynamics' not in p][0], 'rb'))
    rows = [('none', 0.0, b0)]
    frag = 'windows_hz50' if 'bike' in [p for p in glob.glob(os.path.join(D, '*.pkl')) if frag_pred(p)][0] else 'motionsense'
    for p in sorted(glob.glob(os.path.join(D, '*deg*.pkl'))):
        if 'segdetail' in p or 'dynamics' in p or frag not in p:
            continue
        m = re.search(r'_deg(noise|mix)([0-9.]+)\.pkl$', p)
        rows.append((m.group(1), float(m.group(2)), pickle.load(open(p, 'rb'))))
    out = []
    for mode, lev, r in rows:
        fam = load_grid(r, 'probe_activity', 'seen_seen', lams)
        best = np.nanmax(np.stack(list(fam.values())), axis=0)
        p0 = fam['lin'][:, 0]
        valid = float(np.mean(p0)) - c >= 0.05
        out.append((mode, lev, p0, rho_seed(fam['lin'], c) if valid else np.full(3, np.nan),
                    rho_seed(best, c) if valid else np.full(3, np.nan), valid))
    return out


hh = dataset_rows(lambda p: 'windows_hz50_win2.56_ov0.5_bike' in p and 'dlm5' in p, [0, 0.05, 0.1, 0.25, 1], 1 / 3)
ms = dataset_rows(lambda p: 'motionsense' in p and '_cap' not in p, [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1], 1 / 3)

MF.style()
fig, axes = plt.subplots(1, 2, figsize=(MF.DBL_W, 2.6), sharey=True)
mk = {'none': 's', 'noise': 'o', 'mix': '^'}
for ax, (name, rows) in zip(axes, (('HHAR (ambulatory)', hh), ('MotionSense (ambulatory)', ms))):
    for mode, lev, p0, rl, rb, valid in rows:
        if not valid:
            continue
        x = float(np.mean(p0))
        for key, color, off in (('lin', MF.C_BLUE, -0.004), ('best', MF.C_ORANGE, 0.004)):
            v = 100 * (rl if key == 'lin' else rb)
            ax.scatter(np.full(3, x + off), v, s=6, color=color, alpha=0.35, linewidths=0, zorder=2)
            ax.errorbar(x + off, np.nanmean(v), yerr=np.nanstd(v, ddof=1), fmt=mk[mode], color=color,
                        ms=4.2, elinewidth=0.7, capsize=1.5, mec='white', mew=0.5, zorder=3)
    ax.set_title(name, loc='left', fontsize=7.5, pad=3)
    ax.invert_xaxis()
    MF.tidy(ax, xlabel=r'prior decodability (held-out linear probe, $\lambda=0$)')
axes[0].set_ylabel(r'probe-accuracy reduction, $\rho$ (%)')
h = [plt.Line2D([], [], marker='s', color=MF.C_BLUE, ls='', ms=4, label=r'linear ($\rho_{\rm lin}$)'),
     plt.Line2D([], [], marker='s', color=MF.C_ORANGE, ls='', ms=4, label=r'strongest ($\rho_{\rm best}$)'),
     plt.Line2D([], [], marker='s', color=MF.INK_MUTED, ls='', ms=4, mfc='none', label='baseline'),
     plt.Line2D([], [], marker='o', color=MF.INK_MUTED, ls='', ms=4, mfc='none', label='noise'),
     plt.Line2D([], [], marker='^', color=MF.INK_MUTED, ls='', ms=4, mfc='none', label='class mix')]
axes[0].legend(handles=h, frameon=False, loc='upper left', fontsize=6, handletextpad=0.2)
fig.tight_layout(pad=0.4, w_pad=1.0)
for o in ('./figures',):
    os.makedirs(o, exist_ok=True)
    fig.savefig(os.path.join(o, 'fig6_controlled.pdf'))
print('wrote fig6_controlled.pdf (held-out probe, frozen rerun)')
