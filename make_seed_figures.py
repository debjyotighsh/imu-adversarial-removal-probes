"""
fig1_divergence.pdf and fig4_lambda_band.pdf from the frozen rerun
====================================================================
Every point is the mean of three seeds with a +/- SD bar (sample SD, n-1), and the individual
seeds are drawn as small markers. The probe is the HELD-OUT probe: fitted on the training pool
and scored on the evaluation cell of segments the encoder never saw. The discriminator is its
final-epoch training accuracy.

    python make_seed_figures.py --dir gait_experiment_output_v3_frozen
"""
import argparse
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import make_figures as MF
from probe_families_table import find_pkl


def curves(r, probe_key, disc_key):
    ks = sorted(r, key=lambda k: float(k.split('=')[1]))
    lam = [float(k.split('=')[1]) for k in ks]
    probe = np.array([r[k][probe_key + '_raw'] for k in ks], float).T       # (seeds, lambdas)
    disc = np.array([r[k][disc_key + '_raw'] for k in ks], float).T
    return lam, probe, disc


def draw(ax, x, arr, color, ls, marker, label=None, ms=3.8):
    for row in arr:
        ax.scatter(x, row, s=6, color=color, alpha=0.35, linewidths=0, zorder=2)
    ax.errorbar(x, arr.mean(0), yerr=arr.std(0, ddof=1), color=color, ls=ls, marker=marker, ms=ms,
                elinewidth=0.7, capsize=1.5, label=label, zorder=3)


def fig1(r, out):
    lam, probe, disc = curves(r, 'probe_activity_tc_seen_seen', 'disc_act_acc')
    x = np.arange(len(lam))
    fig, ax = plt.subplots(figsize=(MF.COL_W, 2.45))
    draw(ax, x, probe, MF.C_BLUE, '-', 'o', 'post-hoc probe (held-out)')
    draw(ax, x, disc, MF.C_ORANGE, '--', 's', 'discriminator')
    MF.chance_line(ax, 1 / 3)
    ax.annotate('probe', xy=(x[-1], probe.mean(0)[-1]), xytext=(-2, 6), textcoords='offset points',
                ha='right', fontsize=7, color=MF.C_BLUE)
    ax.annotate('discriminator', xy=(x[-1], disc.mean(0)[-1]), xytext=(-6, 7), textcoords='offset points',
                ha='right', fontsize=7, color=MF.C_ORANGE)
    i = lam.index(1)
    ax.annotate('', xy=(x[i], probe.mean(0)[i]), xytext=(x[i], disc.mean(0)[i]),
                arrowprops=dict(arrowstyle='<->', lw=0.7, color=MF.INK_MUTED))
    gap = (probe.mean(0)[i] - disc.mean(0)[i]) * 100
    ax.annotate(f'{gap:.1f} pts', xy=(x[i], (probe.mean(0)[i] + disc.mean(0)[i]) / 2), xytext=(-4, 0),
                textcoords='offset points', ha='right', va='center', fontsize=6.5, color=MF.INK)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{v:g}' for v in lam])
    ax.set_ylim(0.25, 1.04)
    MF.tidy(ax, ylabel='accuracy', xlabel=r'adversarial weight $\lambda$')
    ax.legend(loc='center', bbox_to_anchor=(0.5, 0.7), frameon=False, handlelength=2.2)
    fig.savefig(out)
    plt.close(fig)


def fig4(panels, out):
    fig, axes = plt.subplots(2, 2, figsize=(MF.DBL_W, 3.5), sharey=True)
    for ax, (name, chance, r, pk, dk) in zip(axes.ravel(), panels):
        lam, probe, disc = curves(r, pk, dk)
        x = np.arange(len(lam))
        draw(ax, x, probe, MF.C_BLUE, '-', 'o', 'probe (held-out)', ms=3.4)
        draw(ax, x, disc, MF.C_ORANGE, '--', 's', 'discriminator', ms=3.4)
        MF.chance_line(ax, chance, label=False)
        if 1 in lam:
            ax.axvline(lam.index(1), color=MF.INK_MUTED, lw=0.6, ls=(0, (4, 3)), alpha=0.8, zorder=1)
            ax.annotate(r'$\lambda{=}1$', xy=(lam.index(1), 1.0), xytext=(3, -2),
                        textcoords='offset points', fontsize=6, color=MF.INK_MUTED, va='top')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{v:g}' for v in lam], fontsize=6)
        ax.set_ylim(0.15, 1.06)
        MF.tidy(ax)
        ax.set_title(name, fontsize=7.5, color=MF.INK, pad=3)
    for ax in axes[1]:
        ax.set_xlabel(r'adversarial weight $\lambda$')
    for ax in axes[:, 0]:
        ax.set_ylabel('accuracy')
    axes[0, 0].legend(loc='center right', bbox_to_anchor=(0.92, 0.55), frameon=True,
                      framealpha=0.92, edgecolor='none', handlelength=2.2, fontsize=6.5)
    fig.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.2)
    fig.savefig(out)
    plt.close(fig)


def load(d, frag, pred):
    # five-seed merged run where one exists (HHAR central, RealWorld full pool), as in the paper's figures
    hits = find_pkl('gait_experiment_output_v3_frozen_merged5', frag, pred) or find_pkl(d, frag, pred)
    return pickle.load(open(hits[0], 'rb'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='gait_experiment_output_v3_frozen')
    a = ap.parse_args()
    MF.style()
    hh = load(a.dir, 'windows_hz50_win2.56_ov0.5_bike', lambda b: '_cap' not in b and 'dlm5' in b and '_deg' not in b)
    rw = load(a.dir, 'results_sweep_realworld', lambda b: '_cap' not in b)
    ms = load(a.dir, 'results_sweep_motionsense', lambda b: '_cap' not in b and '_deg' not in b)
    mb = load(a.dir, 'results_sweep_mobiact', lambda b: '_cap' not in b)
    panels = [('HHAR (ambulatory), activity', 1 / 3, hh, 'probe_activity_tc_seen_seen', 'disc_act_acc'),
              ('RealWorld (full pool), activity', 1 / 3, rw, 'probe_activity_tc_seen_seen', 'disc_act_acc'),
              ('MotionSense (ambulatory), activity', 1 / 3, ms, 'probe_activity_tc_seen_seen', 'disc_act_acc'),
              ('MobiAct (stairs up / down), activity', 1 / 2, mb, 'probe_activity_tc_seen_seen', 'disc_act_acc')]
    for o in ('./figures',):
        os.makedirs(o, exist_ok=True)
        fig1(hh, os.path.join(o, 'fig1_divergence.pdf'))
        fig4(panels, os.path.join(o, 'fig4_lambda_band.pdf'))
    print('wrote fig1_divergence.pdf and fig4_lambda_band.pdf')


if __name__ == '__main__':
    main()
