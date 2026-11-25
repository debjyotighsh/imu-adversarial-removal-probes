"""
Regenerate the data figures of the paper (Figures 2-7) from the stored results, with Times New Roman text.
==========================================================================================================
Figure 1 is the TikZ overview diagram (paper_figures/overview).  The six plots are drawn by the existing figure scripts
(make_seed_figures.py, make_probe_figures.py, plot_dynamics.py, plot_restart_dynamics.py, probe_gap_figure.py); this driver
imports them unchanged, applies the Times typeface and sends the output to --out.

Run it from the repository root after unzipping results.zip (the stored results are read from gait_experiment_output_*/):

    python make_times_figures.py --out paper_figures_regenerated

Options:
    --control   keep the original DejaVu fonts and sizes (reproduces the first version of the figures; used to check that this
                driver draws exactly the same data as the original scripts)

What the Times version changes: the typeface only, plus +1 pt on every text size (Times is optically smaller than DejaVu Serif)
and matplotlib's tick-density rule is evaluated at the old sizes so tick positions stay put.  Times New Roman is used when it is
installed; otherwise STIX, which is Times-compatible and ships with matplotlib.  Data, colours, markers, axes and layout come
from the original scripts.

HHAR and RealWorld panels use the five-seed merged run (gait_experiment_output_v3_frozen_merged5) where it exists and the
three-seed frozen run elsewhere, as in the paper.
"""
import argparse
import os
import pickle
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

ap = argparse.ArgumentParser(description=__doc__.split('\n')[1], formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--out', default='paper_figures_regenerated')
ap.add_argument('--control', action='store_true', help='original DejaVu fonts and sizes')
ARGS = ap.parse_args()
OUT = os.path.abspath(ARGS.out)
os.makedirs(OUT, exist_ok=True)

STATS = 'gait_experiment_output_v3_frozen'
DYNAMICS = os.path.join('gait_experiment_output_v2_dynamics', '*_dynamics.pkl')
RESTART = os.path.join('gait_experiment_output_v3_restart_dynamics', '*_dynamics.pkl')
if not os.path.isdir(STATS):
    sys.exit('Cannot find %s/ here. Run from the repository root after unzipping results.zip.' % STATS)

BUMP = 1.0
TIMES_RC = None
if not ARGS.control:
    try:
        fm.findfont('Times New Roman', fallback_to_default=False)
        TIMES_RC = {'font.family': 'serif', 'font.serif': ['Times New Roman'],
                    'mathtext.fontset': 'custom', 'mathtext.rm': 'Times New Roman',
                    'mathtext.it': 'Times New Roman:italic', 'mathtext.bf': 'Times New Roman:bold',
                    'mathtext.default': 'it', 'mathtext.fallback': 'stix',
                    'pdf.fonttype': 42, 'ps.fonttype': 42}
    except ValueError:
        print('Times New Roman is not installed here; using STIX (Times-compatible, bundled with matplotlib).')
        TIMES_RC = {'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'mathtext.fontset': 'stix',
                    'pdf.fonttype': 42, 'ps.fonttype': 42}

if TIMES_RC is not None:
    _orig_set_size = fm.FontProperties.set_size

    def _set_size(self, size):
        """Every text size goes through here (rc defaults, explicit fontsize=, relative names): add BUMP once."""
        if size is None:
            size = plt.rcParams['font.size']
        try:
            val = float(size)
        except (TypeError, ValueError):
            val = fm.font_scalings[size] * fm.FontManager.get_default_size()
        _orig_set_size(self, val + BUMP)

    fm.FontProperties.set_size = _set_size

    # matplotlib picks the number of axis ticks from the tick-label size; measure it before the +BUMP so the tick
    # positions stay where the original figures have them.
    import matplotlib.axis as _maxis
    _orig_tick_label_size = _maxis.Axis._get_tick_label_size

    def _tick_label_size(self, axis_name):
        return _orig_tick_label_size(self, axis_name) - BUMP

    _maxis.Axis._get_tick_label_size = _tick_label_size
    plt.rcParams.update(TIMES_RC)

import make_figures as MF
import make_probe_figures as MPF
import make_seed_figures as MSF
import plot_dynamics as PD
import plot_restart_dynamics as PRD

if TIMES_RC is not None:
    _orig_style = MF.style

    def _style():
        _orig_style()
        plt.rcParams.update(TIMES_RC)

    MF.style = _style                       # make_seed_figures and make_probe_figures call MF.style()
    _orig_pd_style = PD.style

    def _pd_style():
        _orig_pd_style()
        plt.rcParams.update(TIMES_RC)

    PD.style = _pd_style                    # plot_dynamics.main()
    PRD.style = _pd_style                   # plot_restart_dynamics imported its own binding


def load_pref_merged(frag, pred):
    """Five-seed merged run where one exists, else the frozen run (the rule behind the paper's figures)."""
    return pickle.load(open(MPF.hits_for(STATS, frag, pred)[0], 'rb'))


# ---- Figures 2 and 7: discriminator vs probe over lambda (fig1_divergence, fig4_lambda_band)
MF.style()
hh = load_pref_merged('windows_hz50_win2.56_ov0.5_bike', lambda b: '_cap' not in b and 'dlm5' in b and '_deg' not in b)
rw = load_pref_merged('results_sweep_realworld', lambda b: '_cap' not in b)
ms = load_pref_merged('results_sweep_motionsense', lambda b: '_cap' not in b and '_deg' not in b)
mb = load_pref_merged('results_sweep_mobiact', lambda b: '_cap' not in b)
pk, dk = 'probe_activity_tc_seen_seen', 'disc_act_acc'
MSF.fig1(hh, os.path.join(OUT, 'fig1_divergence.pdf'))
MSF.fig4([('HHAR (ambulatory), activity', 1 / 3, hh, pk, dk),
          ('RealWorld (full pool), activity', 1 / 3, rw, pk, dk),
          ('MotionSense (ambulatory), activity', 1 / 3, ms, pk, dk),
          ('MobiAct (stairs up / down), activity', 1 / 2, mb, pk, dk)],
         os.path.join(OUT, 'fig4_lambda_band.pdf'))
print('wrote fig1_divergence.pdf, fig4_lambda_band.pdf')

# ---- Figure 6: training-set size (fig3_datasize)
MPF.fig3(STATS, [OUT])
print('wrote fig3_datasize.pdf')

# ---- Figure 4: training dynamics (fig5_dynamics)
sys.argv = ['plot_dynamics.py', '--pkl', DYNAMICS, '--out_dir', OUT]
PD.main()

# ---- Figure 5: restart dynamics (fig7_restart)
sys.argv = ['plot_restart_dynamics.py', '--pkl', RESTART, '--out_dir', OUT]
PRD.main()

# ---- Figure 3: probe-discriminator gap (fig_probe_gap); this script is run on matplotlib's own defaults
matplotlib.rcdefaults()
if TIMES_RC is not None:
    plt.rcParams.update(TIMES_RC)
sys.argv = ['probe_gap_figure.py', os.path.join(OUT, 'fig_probe_gap.pdf')]
runpy.run_path(os.path.join(HERE, 'probe_gap_figure.py'), run_name='__main__')
print('done:', OUT)
