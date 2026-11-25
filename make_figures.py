"""
Figure generation for the IEEE paper
=====================================
Emits fig1_divergence.pdf, fig2_decodability.pdf, fig3_datasize.pdf and
fig4_lambda_band.pdf — the filenames paper.tex already references.

    # See the figures immediately using the numbers reported in the paper
    python make_figures.py --demo

    # List the sweep CSVs found, with the condition parsed out of each filename
    python make_figures.py --results_dir ./gait_experiment_output_v2 --list

    # Generate from your own CSVs — easiest form
    python make_figures.py --results_dir ./gait_experiment_output_v2 --auto

    # Or name each condition explicitly (globs allowed)
    python make_figures.py --results_dir ./gait_experiment_output_v2 \
        --hhar    results_sweep_hhar_...acts-bike-stairsdown-stairsup-walk..._pareto.csv \
        --rw_full results_sweep_realworld_...ex-none..._pareto.csv \
        --rw_cap  results_sweep_realworld_..._cap15315_pareto.csv \
        --ms      results_sweep_motionsense_...acts-jog-stairsdown-stairsup-walk..._pareto.csv \
        --mobiact results_sweep_mobiact_...acts-stairsdown-stairsup-walk..._pareto.csv

DESIGN NOTES
------------
- Print, two-column IEEE. Single-column figures are 3.5 in wide, 8 pt serif type,
  vector PDF with TrueType embedding (pdf.fonttype 42) so the publisher's
  preflight does not reject Type 3 fonts.
- Every series carries identity three ways — colour, line style and marker shape —
  so the figures read correctly in greyscale and under colour-vision deficiency.
  Colour alone is never the encoding.
- Palette is the validated categorical order (blue, orange, violet, aqua); worst
  adjacent CVD separation ΔE 24.7, all six checks pass on a light surface.
- Probe and discriminator share one axis because both are accuracies on [0, 1].
  This is deliberately not a dual-axis chart.
- The λ axis uses evenly spaced categorical positions, not a linear scale: λ=0 has
  no place on a log axis, and a linear axis compresses the 0–0.3 region where the
  entire effect lives. Say so in the caption.
- No in-figure titles — IEEE captions carry them.

Requires: matplotlib, numpy, pandas
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- validated categorical palette (light surface) ---------------------------
C_BLUE, C_ORANGE, C_VIOLET, C_AQUA = '#2a78d6', '#eb6834', '#4a3aa7', '#1baf7a'
INK, INK_MUTED, GRID = '#0b0b0b', '#52514e', '#c9c9c4'

COL_W, DBL_W = 3.5, 7.16          # IEEE column widths, inches


def style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times New Roman'],
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'legend.fontsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'axes.edgecolor': INK_MUTED,
        'axes.linewidth': 0.6,
        'axes.labelcolor': INK,
        'text.color': INK,
        'xtick.color': INK_MUTED,
        'ytick.color': INK_MUTED,
        'grid.color': GRID,
        'grid.linewidth': 0.4,
        'lines.linewidth': 1.3,
        'lines.markersize': 3.6,
        'figure.dpi': 200,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


def tidy(ax, ylabel=None, xlabel=None):
    """Recessive grid and axes; the data carries the ink."""
    ax.grid(True, axis='y', alpha=0.7)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)


def chance_line(ax, c, label=True, x=0.02):
    ax.axhline(c, color=INK_MUTED, lw=0.7, ls=(0, (1, 2)), zorder=1)
    if label:
        ax.annotate('chance', xy=(x, c), xycoords=('axes fraction', 'data'),
                    xytext=(0, 3), textcoords='offset points',
                    fontsize=6, color=INK_MUTED)


# =============================================================================
# Data: from CSV, or the values reported in the paper (--demo)
# =============================================================================
DEMO = {
    # HHAR ambulatory: bike / stairsdown / stairsup seen, walk held out,
    # s3mini held out, s3mini_2 excluded, 15,315 training windows.
    'hhar': dict(
        lam=[0, 0.02, 0.05, 0.1, 0.15, 0.25, 0.5, 1],
        probe=[0.991, 0.982, 0.989, 0.990, 0.991, 0.990, 0.992, 0.990],
        disc=[0.998, 0.803, 0.462, 0.429, 0.393, 0.384, 0.372, 0.372],
        chance=1 / 3, name='HHAR (ambulatory)'),
    # RealWorld ambulatory, waist held out, full training pool (102,898 windows).
    'rw_full': dict(
        lam=[0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 1, 2],
        probe=[0.853, 0.834, 0.683, 0.522, 0.525, 0.528, 0.563, 0.572,
               0.666, 0.741, 0.803],
        disc=[0.952, 0.947, 0.850, 0.666, 0.609, 0.562, 0.533, 0.508,
              0.488, 0.432, 0.410],
        chance=1 / 3, name='RealWorld (full)'),
    # Same, subsampled to 15,307 windows.
    'rw_cap': dict(
        lam=[0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1],
        probe=[0.803, 0.730, 0.687, 0.668, 0.699, 0.702, 0.748, 0.767],
        disc=[0.970, 0.951, 0.718, 0.668, 0.583, 0.542, 0.483, 0.429],
        chance=1 / 3, name='RealWorld (15.3k)'),
    # MotionSense ambulatory, walk held out, 5,234 training windows.
    'ms': dict(
        lam=[0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1],
        probe=[0.883, 0.757, 0.853, 0.865, 0.845, 0.882, 0.862, 0.896],
        disc=[0.985, 0.595, 0.489, 0.416, 0.508, 0.422, 0.409, 0.401],
        chance=1 / 3, name='MotionSense (ambulatory)'),
    # MobiAct condition B: stairs up / stairs down seen, walk held out.
    'mobiact': dict(
        lam=[0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1],
        probe=[0.934, 0.861, 0.847, 0.849, 0.853, 0.852, 0.865, 0.898],
        disc=[0.956, 0.855, 0.745, 0.698, 0.680, 0.647, 0.575, 0.546],
        chance=0.5, name='MobiAct (stairs)'),
}

# Figure 2: prior decodability against fraction removed. `matched` marks the
# conditions run at comparable training size.
# (label, prior decodability, fraction removed, (dx, dy) label offset in points)
# Offsets are hand-set: six labels in a small panel collide under any automatic
# placement rule.
DECODABILITY = [
    ('MobiAct jog|stand',   1.000, 0.002, (14, 10)),
    ('HHAR ambulatory',     0.991, 0.013, (30, -11)),
    ('MobiAct stairs',      0.934, 0.200, (0, 9)),
    ('MotionSense all',     0.907, 0.259, (2, 9)),
    ('MotionSense ambul.',  0.883, 0.231, (6, -13)),
    ('RealWorld capped',    0.803, 0.288, (-8, 9)),
]


# Which condition each figure needs, expressed as a predicate over the parsed
# filename index. Deliberately explicit: an automatic choice that is wrong is
# worse than one you had to type, so every selection is printed for checking.
AUTO_RULES = {
    'hhar':    lambda r: r.dataset == 'hhar' and 'bike' in r.acts and r.cap == '-',
    'rw_full': lambda r: r.dataset == 'realworld' and r.cap == '-',
    'rw_cap':  lambda r: r.dataset == 'realworld' and r.cap != '-',
    'ms':      lambda r: r.dataset == 'motionsense' and r.cap == '-',
    'mobiact': lambda r: r.dataset == 'mobiact' and r.cap not in ('-', '3720'),
}

AUTO_DESC = {
    'hhar':    'HHAR, ambulatory activities, uncapped',
    'rw_full': 'RealWorld, full training pool',
    'rw_cap':  'RealWorld, subsampled',
    'ms':      'MotionSense, ambulatory, uncapped',
    'mobiact': 'MobiAct, stairs condition (capped, not the 3720 matched run)',
}


def auto_select(results_dir, overrides=None):
    """
    Pick each condition by rule, except where the caller named one explicitly.
    Mixing matters in practice: several conditions select cleanly while one is
    genuinely ambiguous (two held-out positions, say), and re-typing all five
    filenames to disambiguate one of them is how mistakes get made.
    """
    overrides = {k: v for k, v in (overrides or {}).items() if v}
    idx = index_results(results_dir)
    if idx.empty:
        raise SystemExit(f"No *_pareto.csv under {results_dir}")
    chosen = {}
    for key, rule in AUTO_RULES.items():
        if key in overrides:
            chosen[key] = resolve(overrides[key], results_dir)
            continue
        hits = [r.file for r in idx.itertuples() if rule(r)]
        if not hits:
            raise SystemExit(
                f"--auto found no CSV for '{key}' ({AUTO_DESC[key]}).\n"
                f"Run with --list and pass --{key} explicitly.")
        if len(hits) > 1:
            raise SystemExit(
                f"--auto found {len(hits)} candidates for '{key}' "
                f"({AUTO_DESC[key]}). Keep --auto and add --{key} to pin just "
                f"this one:\n  " + "\n  ".join(hits))
        chosen[key] = os.path.join(results_dir, hits[0])
    print("Selected (check these before trusting the figures):")
    for k in AUTO_RULES:
        mark = ' [explicit]' if k in overrides else ''
        print(f"  {k:8s} {AUTO_DESC[k]}{mark}\n           "
              f"{os.path.basename(chosen[k])}")
    return chosen


def resolve(pattern, results_dir):
    """
    Accept a full path, a bare filename, or a glob. These filenames are long
    enough that typing them is error-prone, so a pattern such as
    "*mobiact*cap15737*" is the practical way to select one.
    """
    if os.path.isabs(pattern) and os.path.exists(pattern):
        return pattern
    direct = os.path.join(results_dir, pattern)
    if os.path.exists(direct):
        return direct
    hits = sorted(glob.glob(direct)) or sorted(glob.glob(pattern))
    if not hits:
        raise SystemExit(f"No CSV matched '{pattern}' under {results_dir}. "
                         f"Run with --list to see what is there.")
    if len(hits) > 1:
        raise SystemExit(f"'{pattern}' matched {len(hits)} files; narrow it:\n  " +
                         "\n  ".join(os.path.basename(h) for h in hits))
    return hits[0]


def load_csv(path, chance, name):
    df = pd.read_csv(path).sort_values('lambda_max')
    return dict(lam=df['lambda_max'].tolist(),
                probe=df['probe_activity'].tolist(),
                disc=df['disc_act_acc'].tolist(),
                chance=chance, name=name)


def index_results(results_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(results_dir, '*_pareto.csv'))):
        b = os.path.basename(p)
        rows.append(dict(
            file=b,
            dataset=(re.search(r'results_sweep_([a-z]+)_', b) or ['', '?'])[1],
            acts=(re.search(r'acts-([a-z\-]+)_', b) or ['', '?'])[1],
            heldout=(re.search(r'dev-([A-Za-z0-9_]+?)_act', b) or ['', '?'])[1],
            cap=(re.search(r'_cap(\d+)', b) or ['', '-'])[1],
            mode=(re.search(r'_(adversarial|auxiliary)_', b) or ['', '?'])[1]))
    return pd.DataFrame(rows)


# =============================================================================
# Figures
# =============================================================================
def fig1_divergence(d, out):
    """The paper's central claim: discriminator at chance, probe at ceiling."""
    fig, ax = plt.subplots(figsize=(COL_W, 2.35))
    x = np.arange(len(d['lam']))

    ax.plot(x, d['probe'], color=C_BLUE, ls='-', marker='o',
            label='post-hoc probe', zorder=3)
    ax.plot(x, d['disc'], color=C_ORANGE, ls='--', marker='s',
            label='discriminator', zorder=3)
    chance_line(ax, d['chance'])

    # Direct labels: two series, so both are named on the plot as well as in the
    # legend — identity is never colour-alone.
    ax.annotate('probe', xy=(x[-1], d['probe'][-1]), xytext=(-2, 6),
                textcoords='offset points', ha='right', fontsize=7, color=C_BLUE)
    ax.annotate('discriminator', xy=(x[-1], d['disc'][-1]), xytext=(-6, 7),
                textcoords='offset points', ha='right', fontsize=7,
                color=C_ORANGE)

    # Annotate the divergence at the field's default lambda.
    if 1 in d['lam']:
        i = d['lam'].index(1)
        ax.annotate('', xy=(x[i], d['probe'][i]), xytext=(x[i], d['disc'][i]),
                    arrowprops=dict(arrowstyle='<->', lw=0.7, color=INK_MUTED))
        gap = (d['probe'][i] - d['disc'][i]) * 100
        ax.annotate(f'{gap:.1f} pts', xy=(x[i], (d['probe'][i] + d['disc'][i]) / 2),
                    xytext=(-4, 0), textcoords='offset points',
                    ha='right', va='center', fontsize=6.5, color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{v:g}' for v in d['lam']])
    ax.set_ylim(0.25, 1.04)
    tidy(ax, ylabel='accuracy', xlabel=r'adversarial weight $\lambda$')
    ax.legend(loc='center left', frameon=False, handlelength=2.2)
    fig.savefig(out)
    plt.close(fig)
    return out


def fig2_decodability(points, out):
    """One series, direct-labelled — no legend box needed."""
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    xs = [p[1] for p in points]
    ys = [p[2] * 100 for p in points]

    ax.plot(xs, ys, color=GRID, lw=0.8, ls='-', zorder=1)
    ax.scatter(xs, ys, s=26, color=C_BLUE, marker='o',
               edgecolor='white', linewidth=0.6, zorder=3)

    for name, x, y, (dx, dy) in points:
        ax.annotate(name, xy=(x, y * 100), xytext=(dx, dy),
                    textcoords='offset points', ha='center',
                    fontsize=6.2, color=INK_MUTED)

    ax.invert_xaxis()
    ax.set_ylim(-8, 42)
    ax.set_xlim(1.02, 0.77)
    tidy(ax, ylabel=r'probe-accuracy reduction, $\rho$ (%)',
         xlabel=r'prior decodability (probe accuracy at $\lambda=0$)')
    fig.savefig(out)
    plt.close(fig)
    return out


def fig3_datasize(full, cap, out):
    """
    The two sweeps do not share a lambda grid, so each series is positioned by
    its OWN lambda values on the union grid. Plotting both against their own
    array index would silently draw the shorter series against the wrong
    lambdas.
    """
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))

    grid = sorted(set(full['lam']) | set(cap['lam']))
    pos = {v: i for i, v in enumerate(grid)}

    for d, color, ls, mk, lab in ((full, C_BLUE, '-', 'o', '102,898 windows'),
                                  (cap, C_VIOLET, '-.', '^', '15,307 windows')):
        xs = [pos[v] for v in d['lam']]
        ax.plot(xs, d['probe'], color=color, ls=ls, marker=mk, label=lab,
                zorder=3)
    chance_line(ax, full['chance'], x=0.62)

    ax.set_xticks(range(len(grid)))
    ax.set_xticklabels([f'{v:g}' for v in grid])
    ax.set_ylim(0.28, 0.97)
    tidy(ax, ylabel='activity probe accuracy',
         xlabel=r'adversarial weight $\lambda$')
    ax.legend(loc='upper center', frameon=False, handlelength=2.4, ncol=1)
    fig.savefig(out)
    plt.close(fig)
    return out


def fig4_band(panels, out):
    """Small multiples: four datasets rather than eight lines on one axis."""
    fig, axes = plt.subplots(2, 2, figsize=(DBL_W, 3.5), sharey=True)
    for ax, d in zip(axes.ravel(), panels):
        x = np.arange(len(d['lam']))
        ax.plot(x, d['probe'], color=C_BLUE, ls='-', marker='o', label='probe')
        ax.plot(x, d['disc'], color=C_ORANGE, ls='--', marker='s',
                label='discriminator')
        chance_line(ax, d['chance'], label=False)

        # Mark the field's default lambda, which our results place outside the
        # effective band on every dataset.
        if 1 in d['lam']:
            ax.axvline(d['lam'].index(1), color=INK_MUTED, lw=0.6,
                       ls=(0, (4, 3)), alpha=0.8, zorder=1)
            ax.annotate(r'$\lambda{=}1$', xy=(d['lam'].index(1), 1.0),
                        xytext=(3, -2), textcoords='offset points',
                        fontsize=6, color=INK_MUTED, va='top')

        ax.set_xticks(x)
        ax.set_xticklabels([f'{v:g}' for v in d['lam']], fontsize=6)
        ax.set_ylim(0.15, 1.06)
        tidy(ax)
        ax.set_title(d['name'], fontsize=7.5, color=INK, pad=3)

    for ax in axes[1]:
        ax.set_xlabel(r'adversarial weight $\lambda$')
    for ax in axes[:, 0]:
        ax.set_ylabel('accuracy')
    axes[0, 0].legend(loc='center left', frameon=True, framealpha=0.92,
                      edgecolor='none', handlelength=2.2, fontsize=6.5)
    fig.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.2)
    fig.savefig(out)
    plt.close(fig)
    return out


# =============================================================================
def main():
    p = argparse.ArgumentParser(description="Generate the paper's figures")
    p.add_argument('--results_dir', default='./gait_experiment_output_v2')
    p.add_argument('--out_dir', default='./figures')
    p.add_argument('--demo', action='store_true',
                   help="Use the values reported in the paper instead of reading "
                        "CSVs. Good for checking layout before wiring paths.")
    p.add_argument('--list', action='store_true',
                   help="Index the sweep CSVs found and exit.")
    p.add_argument('--auto', action='store_true',
                   help="Pick the five conditions automatically from --results_dir and "
                        "print what was chosen. Avoids typing (or shell-quoting) the long "
                        "filenames. Any condition you also pass explicitly overrides the "
                        "automatic choice, so an ambiguous one can be pinned on its own. "
                        "Verify the printed choices before trusting the figures.")
    for k in ('hhar', 'rw_full', 'rw_cap', 'ms', 'mobiact'):
        p.add_argument(f'--{k}', default=None, help=f'CSV for the {k} condition')
    args = p.parse_args()

    if args.list:
        idx = index_results(args.results_dir)
        if idx.empty:
            print(f"No *_pareto.csv under {args.results_dir}")
        else:
            print(idx.to_string(index=False))
        return

    style()
    os.makedirs(args.out_dir, exist_ok=True)

    data = {k: dict(DEMO[k]) for k in DEMO}
    if args.auto:
        overrides = {k: getattr(args, k) for k in DEMO}
        for k, path in auto_select(args.results_dir, overrides).items():
            data[k] = load_csv(path, DEMO[k]['chance'], DEMO[k]['name'])
    elif not args.demo:
        missing = [k for k in DEMO if getattr(args, k) is None]
        if missing:
            raise SystemExit(
                "Provide a CSV for each condition, or pass --auto (pick them "
                "automatically) or --demo.\n"
                f"Missing: {', '.join('--' + m for m in missing)}\n"
                "Run with --list to see which sweep CSVs you have.")
        for k in DEMO:
            path = resolve(getattr(args, k), args.results_dir)
            data[k] = load_csv(path, DEMO[k]['chance'], DEMO[k]['name'])
            print(f"  {k:8s} <- {os.path.basename(path)}")

    o = args.out_dir
    print(fig1_divergence(data['hhar'], os.path.join(o, 'fig1_divergence.pdf')))
    print(fig2_decodability(DECODABILITY, os.path.join(o, 'fig2_decodability.pdf')))
    print(fig3_datasize(data['rw_full'], data['rw_cap'],
                        os.path.join(o, 'fig3_datasize.pdf')))
    print(fig4_band([data['hhar'], data['rw_full'], data['ms'], data['mobiact']],
                    os.path.join(o, 'fig4_lambda_band.pdf')))

    if args.demo:
        print("\n  NOTE: --demo used the values transcribed into this script. "
              "Regenerate from\n  your CSVs before submission so the figures and "
              "the tables cannot drift apart.")


if __name__ == '__main__':
    main()
