"""Figure for the probe-generalisation gap: G = strongest-style probe accuracy - held-out discriminator balanced accuracy,
at lambda = 1 (and lambda = 0 as the reference), along four axes. Markers: seed means; thin points: seeds."""
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import probe_gap_report as P

out = sys.argv[1] if len(sys.argv) > 1 else 'fig_probe_gap.pdf'
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(2, 2, figsize=(7.0, 3.9))


def Gseed(r, lam, key):
    return np.asarray(r[f'lambda={lam:g}'][key + '_raw'], float) - np.asarray(r[f'lambda={lam:g}']['disc_act_heldout_tc_seen_seen_raw'], float)


def draw(a, xs, ys, label, color, marker, ls='-'):
    m = [np.nanmean(y) for y in ys]
    a.plot(xs, m, marker=marker, color=color, ls=ls, label=label, ms=4, lw=1.1)
    for x, y in zip(xs, ys):
        a.scatter([x] * len(y), y, s=6, color=color, alpha=0.4, zorder=1)


runs = {'HHAR': (P.load('hhar_ep25'), 1.0, '#1b6ca8'), 'MobiAct B': (P.load('mobiact_ep25'), 1.0, '#c0504d')}

# (a) probe capacity: MLP hidden width, linear C, k-NN k  (x axis = ordinal position)
a = ax[0, 0]
groups = [('lr_C0.001', 'lr_C0.1', 'lr_C10'), ('mlp_w4', 'mlp_w16', 'mlp_w64', 'mlp_w256', 'mlp_w1024'), ('knn_k1', 'knn_k5', 'knn_k50')]
labels = ['LR C=1e-3', 'LR C=0.1', 'LR C=10', 'MLP 4', 'MLP 16', 'MLP 64', 'MLP 256', 'MLP 1024', 'kNN 1', 'kNN 5', 'kNN 50']
keys = [f'probe_activity_tc_seen_seen_grid_{n}' for g in groups for n in g]
for name, (r, lam, col) in runs.items():
    ys = [Gseed(r, lam, k) for k in keys]
    draw(a, np.arange(len(keys)), ys, name, col, {'HHAR': 'o', 'MobiAct B': 's'}[name], ls={'HHAR': '-', 'MobiAct B': '--'}[name])
a.set_xticks(range(len(keys)))
a.set_xticklabels(labels, rotation=60, ha='right', fontsize=6.5)
a.set_ylabel('gap $G$'); a.set_ylim(0, 0.75); a.set_title('(a) probe capacity ($\\lambda=1$)', fontsize=8)
a.legend(frameon=False, fontsize=7, loc='lower right')

# (b) probe fit-set size
a = ax[0, 1]
sizes = [100, 300, 1000, 3000]
for name, (r, lam, col) in runs.items():
    for fam, mk in (('lr', 'o'), ('mlp', 's')):               # family by marker, dataset by line style
        ys = [Gseed(r, lam, f'probe_activity_tc_seen_seen_grid_fit{n}_{fam}') for n in sizes]
        draw(a, sizes, ys, f'{name} {fam.upper()}', col, mk, {'HHAR': '-', 'MobiAct B': '--'}[name])
a.set_xscale('log'); a.set_ylim(0, 0.75); a.set_xlabel('probe fit windows'); a.set_ylabel('gap $G$')
a.set_title('(b) probe fit-set size ($\\lambda=1$)', fontsize=8); a.legend(frameon=False, fontsize=6, ncol=2, loc='lower right')

# (c) encoder training windows, (d) epochs: strongest-probe G at lambda = 0 and 1
def strongestG(name, lam):
    r = P.load(name)
    d = np.asarray(r[f'lambda={lam:g}']['disc_act_heldout_tc_seen_seen_raw'], float)
    return P.strongest(r, lam) - d

a = ax[1, 0]
ns = [1000, 3000, 15315]; names = ['hhar_n1000', 'hhar_n3000', 'hhar_ep25']
for lam, col, ls in ((1.0, '#1b6ca8', '-'), (0.0, '0.4', '--')):
    draw(a, ns, [strongestG(n, lam) for n in names], f'$\\lambda={lam:g}$', col, 'o', ls)
a.set_xscale('log'); a.set_ylim(-0.02, 0.75); a.set_xlabel('encoder training windows (25 epochs)'); a.set_ylabel('gap $G$')
a.set_title('(c) encoder training-set size (HHAR)', fontsize=8); a.legend(frameon=False, fontsize=7, loc='upper left')

a = ax[1, 1]
es = [5, 10, 25, 50, 100]
for lam, col, ls in ((1.0, '#1b6ca8', '-'), (0.0, '0.4', '--')):
    draw(a, es, [strongestG(f'hhar_ep{e}', lam) for e in es], f'$\\lambda={lam:g}$', col, 'o', ls)
a.set_xscale('log'); a.set_xticks(es); a.set_xticklabels([str(e) for e in es]); a.set_ylim(-0.02, 0.75)
a.set_xlabel('epochs'); a.set_ylabel('gap $G$'); a.set_title('(d) optimisation budget (HHAR)', fontsize=8)
a.legend(frameon=False, fontsize=7, loc='center right')
fig.tight_layout()
fig.savefig(out)
print('saved', out)

