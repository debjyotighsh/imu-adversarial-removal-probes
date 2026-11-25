"""
Restart-condition dynamics figure (fig7_restart.pdf): diagnoses why the discriminator's
in-training accuracy plateaus well below the post-hoc probe's even when restarts stop it
from collapsing to a constant predictor (Section sec:restart).

  (a) discriminator training accuracy per epoch, dashed vertical lines at each restart
      (epoch 5, 10, 15, 20): within every window it is still climbing when cut off,
      never given an uninterrupted run at the task.
  (b) recovery on the FROZEN final (restart-trained) encoder: the same in-training head,
      continued, and a fresh wide head, both against extra epochs once the encoder stops
      moving. Both jump to >0.9 within one epoch, matching the post-hoc probe -- the
      representation was never actually invariant, the in-training head was just always
      chasing a moving target.

    python plot_restart_dynamics.py --pkl gait_experiment_output_v3_restart_dynamics\\*_dynamics.pkl
"""
import argparse
import glob
import os
import pickle

import numpy as np

from plot_dynamics import band, recovery_series, series, style, tidy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

C_BLUE, C_ORANGE, C_AQUA, INK_MUTED = '#2a78d6', '#eb6834', '#1baf7a', '#52514e'
CHANCE = 1 / 3
RESTART_EVERY = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', required=True)
    ap.add_argument('--out_dir', default='./figures')
    args = ap.parse_args()

    path = sorted(glob.glob(args.pkl))[0]
    D = pickle.load(open(path, 'rb'))
    style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.0, 2.55))

    runs = D['lambda=1']
    x, m, s = series(runs, 'act_acc')
    band(a, x, m, s, C_ORANGE, '-', label='discriminator (training)', marker='o')
    for ep in range(RESTART_EVERY, int(x.max()), RESTART_EVERY):
        a.axvline(ep + 0.5, color=INK_MUTED, lw=0.6, ls=(0, (2, 2)), alpha=0.6)
    a.axhline(CHANCE, color=INK_MUTED, lw=0.7, ls=(0, (1, 2)))
    a.annotate('restart', xy=(RESTART_EVERY + 0.5, 0.97), xytext=(2, 0),
               textcoords='offset points', fontsize=6, color=INK_MUTED, rotation=90, va='top')
    a.set_ylim(0.25, 1.04)
    a.legend(frameon=False, loc='lower right', handlelength=2.0)
    tidy(a, 'epoch', 'accuracy', '(a) discriminator, restarts every 5 epochs')

    if runs[0].get('recovery'):
        xr, m, s = recovery_series(runs, 'cont', 'test_bal_acc')
        band(b, xr, m, s, C_ORANGE, '-', label='in-training head, continued', marker='s')
        xr, m, s = recovery_series(runs, 'fresh', 'test_bal_acc')
        band(b, xr, m, s, C_AQUA, '-', label='fresh wide head', marker='^')
        b.axhline(CHANCE, color=INK_MUTED, lw=0.7, ls=(0, (1, 2)))
        b.set_ylim(0.25, 1.04)
        b.legend(frameon=False, loc='lower right', handlelength=2.0)
    tidy(b, 'extra epochs, encoder frozen', 'balanced accuracy (in-pool held-back)', '(b) recovery once training stops')

    fig.tight_layout(pad=0.4, w_pad=1.4)
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, 'fig7_restart.pdf')
    fig.savefig(out)
    print('wrote', out)


if __name__ == '__main__':
    main()
