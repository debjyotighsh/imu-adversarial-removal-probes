"""
Multi-probe figures (fig2_decodability.pdf, fig3_datasize.pdf)
===============================================================
Regenerates the two removability figures from the SAME rerun (three probe
families per run) so that linear and strongest-probe removal are comparable. Both
figures show every seed (small markers / thin lines) together with the mean and
its sample SD (n-1):

  fig2  removal against prior decodability, one point per condition, drawn twice:
        linear removability (rho_lin) and removal by the strongest probe (rho_best),
        each computed per seed
  fig3  RealWorld at two training-set sizes, activity probe accuracy against lambda:
        (a) linear probe, (b) strongest probe (max of linear / MLP / kNN)

    python make_probe_figures.py --dir gait_experiment_output_v2_probes
"""
import argparse
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import make_figures as MF
import seed_stats as SS
from probe_families_table import CONDITIONS, find_pkl

C_BLUE, C_ORANGE, C_VIOLET, INK_MUTED = MF.C_BLUE, MF.C_ORANGE, MF.C_VIOLET, MF.INK_MUTED


MERGED5 = 'gait_experiment_output_v3_frozen_merged5'   # five-seed merge of HHAR central and RealWorld full


def hits_for(d, frag, pred):
    """Prefer the five-seed merged run where one exists (HHAR, RealWorld full pool), as in the paper."""
    return find_pkl(MERGED5, frag, pred) or find_pkl(d, frag, pred)


def load(d, label_prefix):
    for label, frag, pred, factors in CONDITIONS:
        if label.startswith(label_prefix):
            hits = hits_for(d, frag, pred)
            return pickle.load(open(hits[0], 'rb')), factors
    raise KeyError(label_prefix)


def sd(v):
    return float(np.std(v, ddof=1))


def fig2(d, out_paths):
    pts = []
    for label, frag, pred, factors in CONDITIONS:
        if label.startswith('RealWorld (ambulatory, full)'):
            continue                                   # different size class, as in the old figure
        hits = hits_for(d, frag, pred)
        if not hits:
            continue
        r = pickle.load(open(hits[0], 'rb'))
        stem, c = factors['activity']
        a = SS.analyse(r, stem, c, tc='seen_seen')      # held-out probes
        pts.append((label, a['p0'], 100 * a['rho_lin'], 100 * a['rho_best']))   # per-seed arrays

    short = {'HHAR (ambulatory)': 'HHAR', 'RealWorld (ambulatory, 15.3k)': 'RealWorld',
             'MobiAct B (stairs up/down)': 'MobiAct B', 'MobiAct A (jog/stand, 3.6k)': 'MobiAct A*',
             'MotionSense (ambulatory)': 'MotionSense amb.', 'MotionSense (all, 5.2k)': 'MotionSense all'}
    shape = {'HHAR': 'o', 'RealWorld': 's', 'MobiAct B': 'v', 'MobiAct A*': '^',
             'MotionSense amb.': 'D', 'MotionSense all': 'P'}
    MF.style()
    fig, ax = plt.subplots(figsize=(MF.COL_W, 2.7))
    for lab, p0, rl, rb in pts:
        ax.plot([p0.mean(), p0.mean()], [rb.mean(), rl.mean()], color=MF.GRID, lw=0.8, zorder=1)
    for idx, color, name in ((2, C_BLUE, r'linear probe ($\rho_{\rm lin}$)'),
                             (3, C_ORANGE, r'strongest probe ($\rho_{\rm best}$)')):
        for lab, p0, rl, rb in pts:
            mk = shape[short[lab]]
            p = (lab, p0, rl, rb)
            ax.scatter(p[1], p[idx], s=6, color=color, marker=mk, alpha=0.4, linewidths=0, zorder=2)
            ax.errorbar(p[1].mean(), p[idx].mean(), xerr=sd(p[1]), yerr=sd(p[idx]), fmt=mk,
                        ms=4.6, color=color, mec='white', mew=0.5, elinewidth=0.7, capsize=1.5, zorder=3)
    ax.invert_xaxis()
    ax.set_ylim(-4, 20)
    MF.tidy(ax, ylabel=r'probe-accuracy reduction, $\rho$ (%)',
            xlabel=r'prior decodability (linear probe, $\lambda=0$)')
    color_h = [plt.Line2D([], [], marker='o', color=C_BLUE, ls='', ms=4, label=r'linear ($\rho_{\rm lin}$)'),
              plt.Line2D([], [], marker='o', color=C_ORANGE, ls='', ms=4, label=r'strongest ($\rho_{\rm best}$)')]
    shape_h = [plt.Line2D([], [], marker=m, color=INK_MUTED, ls='', ms=4.2, mfc='none', label=n)
              for n, m in shape.items()]
    leg1 = ax.legend(handles=color_h, frameon=False, loc='upper right', handlelength=1.4, fontsize=6.4)
    ax.add_artist(leg1)
    ax.legend(handles=shape_h, frameon=False, loc='upper left', handlelength=1.2, fontsize=5.8,
             labelspacing=0.3, handletextpad=0.4)
    for p in out_paths:
        os.makedirs(os.path.dirname(p) or '.', exist_ok=True)
        fig.savefig(os.path.join(p, 'fig2_decodability.pdf'))
    plt.close(fig)
    return [(p[0], p[1].mean(), p[2].mean(), p[3].mean()) for p in pts]


def fig3(d, out_paths):
    full, fac = load(d, 'RealWorld (ambulatory, full)')
    cap, _ = load(d, 'RealWorld (ambulatory, 15.3k)')
    stem = fac['activity'][0]
    data = {}
    for name, r in (('full', full), ('cap', cap)):
        lam, P = SS.seed_curves(r, stem, 'seen_seen')   # held-out probes
        best = np.nanmax(np.stack(list(P.values())), axis=0)          # (seeds, lambdas)
        data[name] = (lam, P['lin'], best)
    grid = sorted(set(data['full'][0]) | set(data['cap'][0]))
    pos = {v: i for i, v in enumerate(grid)}
    MF.style()
    fig, axes = plt.subplots(1, 2, figsize=(MF.DBL_W, 2.5), sharey=True)
    for ax, idx, title in ((axes[0], 1, '(a) linear probe'), (axes[1], 2, '(b) strongest probe')):
        for name, color, ls, mk, lab in (('full', C_BLUE, '-', 'o', '102,898 windows'),
                                         ('cap', C_VIOLET, '-.', '^', '15,307 windows')):
            lam = data[name][0]
            x = [pos[v] for v in lam]
            Y = data[name][idx]                                        # (seeds, lambdas)
            for row in Y:                                              # individual seeds
                ax.plot(x, row, color=color, lw=0.5, alpha=0.35, zorder=2)
                ax.scatter(x, row, s=5, color=color, alpha=0.35, linewidths=0, zorder=2)
            ax.errorbar(x, Y.mean(axis=0), yerr=Y.std(axis=0, ddof=1), color=color, ls=ls, marker=mk,
                        ms=3.6, elinewidth=0.7, capsize=1.5, label=lab, zorder=3)
        ax.axhline(1 / 3, color=INK_MUTED, lw=0.7, ls=(0, (1, 2)), zorder=1)
        ax.set_xticks(range(len(grid)))
        ax.set_xticklabels([f'{v:g}' for v in grid])
        ax.set_ylim(0.28, 0.97)
        MF.tidy(ax, xlabel=r'adversarial weight $\lambda$')
        ax.set_title(title, loc='left', fontsize=7.5, pad=3)
    axes[0].set_ylabel('activity probe accuracy')
    # sits in the free band between the chance line (1/3) and the lowest curve, clear of both
    axes[0].legend(frameon=False, loc='lower center', bbox_to_anchor=(0.62, 0.11), handlelength=2.2)
    fig.tight_layout(pad=0.4, w_pad=1.0)
    for p in out_paths:
        os.makedirs(os.path.dirname(p) or '.', exist_ok=True)
        fig.savefig(os.path.join(p, 'fig3_datasize.pdf'))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='gait_experiment_output_v3_frozen')
    ap.add_argument('--out', nargs='+', default=['./figures'])
    args = ap.parse_args()
    pts = fig2(args.dir, args.out)
    for p in pts:
        print(f"{p[0]:32s} p0={p[1]:.3f}  rho_lin={p[2]:5.1f}%  rho_best={p[3]:5.1f}%")
    fig3(args.dir, args.out)
    print('wrote fig2_decodability.pdf, fig3_datasize.pdf to', args.out)


if __name__ == '__main__':
    main()

