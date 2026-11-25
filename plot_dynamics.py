"""
Training-dynamics figure (fig5_dynamics.pdf)
=============================================
Reads the *_dynamics.pkl written by gait_dual_adversarial_v2.py --track_dynamics
and draws six panels for the HHAR central condition, mean +/- std over seeds:

  (a) activity discriminator training accuracy vs. epoch, with the post-hoc
      linear probe fitted on checkpoints of the same run
  (b) activity discriminator loss vs. epoch
  (c) encoder gradient norm split into its identity part and its adversarial
      (reversed, lambda-scaled) part
  (d) representation drift on a fixed batch of windows, per epoch
  (e) discriminator-head gradient norm
  (f) recovery: an activity head trained on the FROZEN final encoder

    python plot_dynamics.py --pkl gait_experiment_output_v2_dynamics\\*_dynamics.pkl
"""
import argparse
import glob
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

C_BLUE, C_ORANGE, C_VIOLET, C_AQUA, C_GREY = '#2a78d6', '#eb6834', '#4a3aa7', '#1baf7a', '#8a8a85'
INK, INK_MUTED, GRID = '#0b0b0b', '#52514e', '#c9c9c4'
LAM_STYLE = {'lambda=0': (C_GREY, ':', r'$\lambda=0$'),
             'lambda=0.1': (C_VIOLET, '-.', r'$\lambda=0.1$'),
             'lambda=1': (C_ORANGE, '--', r'$\lambda=1$')}
CHANCE = 1 / 3
MAJORITY = 0.3726        # share of the largest activity class (bike) in the HHAR training pool


def style():
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['DejaVu Serif', 'Times New Roman'],
        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 6.5,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.edgecolor': INK_MUTED,
        'axes.linewidth': 0.6, 'axes.labelcolor': INK, 'text.color': INK,
        'xtick.color': INK_MUTED, 'ytick.color': INK_MUTED, 'grid.color': GRID,
        'grid.linewidth': 0.4, 'lines.linewidth': 1.3, 'lines.markersize': 3.4,
        'figure.dpi': 200, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
        'pdf.fonttype': 42, 'ps.fonttype': 42})


def tidy(ax, xlabel=None, ylabel=None, title=None):
    ax.grid(True, axis='y', alpha=0.7)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc='left', fontsize=7.5, pad=3)


def series(runs, field):
    """(epochs, mean, std) of a per-epoch field across seeds; NaNs are skipped."""
    per_seed = []
    for r in runs:
        d = {rec['epoch']: rec.get(field, np.nan) for rec in r['dynamics']}
        per_seed.append(d)
    eps = sorted({e for d in per_seed for e in d})
    arr = np.array([[d.get(e, np.nan) for e in eps] for d in per_seed], float)
    keep = ~np.all(np.isnan(arr), axis=0)
    eps, arr = np.array(eps)[keep], arr[:, keep]
    global _LAST
    _LAST = arr                                    # per-seed values, drawn by band()
    return eps, np.nanmean(arr, 0), np.nanstd(arr, 0, ddof=1)      # sample SD (n-1)


_LAST = None


def band(ax, x, m, s, color, ls, label=None, marker=None):
    """seed mean with a +/- SD band; the individual seeds are drawn as thin lines."""
    if _LAST is not None and _LAST.shape[1] == len(x):
        for row in _LAST:
            ax.plot(x, row, color=color, lw=0.4, alpha=0.35, zorder=2)
    ax.plot(x, m, color=color, ls=ls, marker=marker, label=label, zorder=3)
    ax.fill_between(x, m - s, m + s, color=color, alpha=0.15, lw=0, zorder=2)


def recovery_series(runs, which, field):
    arr = np.array([[c[field] for c in r['recovery'][which]] for r in runs], float)
    x = np.array([c['epoch'] for c in runs[0]['recovery'][which]])
    global _LAST
    _LAST = arr
    return x, arr.mean(0), arr.std(0, ddof=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', required=True)
    ap.add_argument('--out_dir', default='./figures')
    ap.add_argument('--also_copy_to', nargs='*', default=[])
    args = ap.parse_args()

    path = sorted(glob.glob(args.pkl))[0]
    D = pickle.load(open(path, 'rb'))
    style()
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.3))
    (a, b, c), (d, e, f) = axes

    # (a) discriminator vs probe, lambda = 1 (plus lambda = 0 discriminator)
    for key in ('lambda=0', 'lambda=1'):
        if key in D:
            col, ls, lab = LAM_STYLE[key]
            x, m, s = series(D[key], 'act_acc')
            band(a, x, m, s, col, ls, label='discriminator, ' + lab)
    if 'lambda=1' in D:
        x, m, s = series(D['lambda=1'], 'probe_act')
        band(a, x, m, s, C_BLUE, '-', label=r'post-hoc probe, $\lambda=1$', marker='o')
    a.axhline(CHANCE, color=INK_MUTED, lw=0.7, ls=(0, (1, 2)))
    a.axhline(MAJORITY, color=C_ORANGE, lw=0.6, ls=(0, (3, 2)), alpha=0.6)
    a.annotate('uniform chance', xy=(0.98, CHANCE), xycoords=('axes fraction', 'data'),
               xytext=(0, -7), textcoords='offset points', ha='right', fontsize=6,
               color=INK_MUTED)
    a.annotate('majority-class rate', xy=(0.98, MAJORITY), xycoords=('axes fraction', 'data'),
               xytext=(0, 3), textcoords='offset points', ha='right', fontsize=6,
               color=C_ORANGE)
    a.set_ylim(0.25, 1.04)
    a.legend(frameon=False, loc='center right', bbox_to_anchor=(1.0, 0.62),
             handlelength=2.2)
    tidy(a, 'epoch', 'accuracy', '(a) discriminator vs. probe')

    # (b) discriminator loss
    for key in ('lambda=0', 'lambda=0.1', 'lambda=1'):
        if key in D:
            col, ls, lab = LAM_STYLE[key]
            x, m, s = series(D[key], 'act_loss')
            band(b, x, m, s, col, ls, label=lab)
    b.axhline(np.log(3), color=INK_MUTED, lw=0.7, ls=(0, (1, 2)))
    b.annotate(r'$\ln 3$ (chance)', xy=(0.98, np.log(3)), xycoords=('axes fraction', 'data'),
               xytext=(0, 3), textcoords='offset points', ha='right', fontsize=6,
               color=INK_MUTED)
    b.legend(frameon=False, handlelength=2.2)
    tidy(b, 'epoch', 'cross-entropy', '(b) discriminator loss')

    # (c) encoder gradient split, lambda = 1
    if 'lambda=1' in D:
        x, m, s = series(D['lambda=1'], 'gn_enc_id')
        band(c, x, m, s, C_BLUE, '-', label='identity part')
        x, m, s = series(D['lambda=1'], 'gn_enc_adv_act')
        band(c, x, m, s, C_ORANGE, '--', label='adversarial part')
        c.legend(frameon=False, handlelength=2.2)
    tidy(c, 'epoch', r'encoder $\|\nabla\|_2$', r'(c) encoder gradient, $\lambda=1$')

    # (d) representation drift
    for key in ('lambda=0', 'lambda=0.1', 'lambda=1'):
        if key in D:
            col, ls, lab = LAM_STYLE[key]
            x, m, s = series(D[key], 'rep_drift')
            band(d, x, m, s, col, ls, label=lab)
    d.legend(frameon=False, handlelength=2.2)
    tidy(d, 'epoch', 'relative drift', '(d) representation drift')

    # (e) discriminator-head gradient norm
    for key in ('lambda=0', 'lambda=0.1', 'lambda=1'):
        if key in D:
            col, ls, lab = LAM_STYLE[key]
            x, m, s = series(D[key], 'gn_head_act')
            band(e, x, m, s, col, ls, label=lab)
    e.legend(frameon=False, handlelength=2.2)
    tidy(e, 'epoch', r'head $\|\nabla\|_2$', '(e) discriminator-head gradient')

    # (f) recovery on the frozen final encoder, lambda = 1
    if 'lambda=1' in D and D['lambda=1'][0].get('recovery'):
        runs = D['lambda=1']
        x, m, s = recovery_series(runs, 'cont', 'train_acc')
        band(f, x, m, s, C_ORANGE, '--', label='trained head (fit split)', marker='s')
        x, m, s = recovery_series(runs, 'cont', 'test_acc')
        band(f, x, m, s, C_ORANGE, '-', label='trained head (held out)', marker='s')
        x, m, s = recovery_series(runs, 'fresh', 'test_acc')
        band(f, x, m, s, C_AQUA, '-', label='fresh wide head (held out)', marker='^')
        f.axhline(CHANCE, color=INK_MUTED, lw=0.7, ls=(0, (1, 2)))
        f.set_ylim(0.25, 1.04)
        f.legend(frameon=False, loc='center right', handlelength=2.2)
    tidy(f, 'extra epochs, encoder frozen', 'accuracy', r'(f) recovery, $\lambda=1$')

    fig.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, 'fig5_dynamics.pdf')
    fig.savefig(out)
    print('wrote', out)
    for extra in args.also_copy_to:
        os.makedirs(extra, exist_ok=True)
        fig.savefig(os.path.join(extra, 'fig5_dynamics.pdf'))
        print('wrote', os.path.join(extra, 'fig5_dynamics.pdf'))


if __name__ == '__main__':
    main()
