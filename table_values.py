"""
LaTeX-ready mean +/- SD (sample SD, n-1) values for the manuscript tables.

Removal statistics are computed per seed (seed_stats.py) and summarised over the three seeds;
accuracies for the original single-family sweeps come from the stored (mean, population-SD)
tuples, converted to sample SD.

    python table_values.py
"""
import glob
import os
import pickle

import numpy as np

import seed_stats as SS
from probe_families_table import CONDITIONS, find_pkl

F = np.sqrt(1.5)          # population SD (ddof=0, n=3) -> sample SD (ddof=1)


def pm(m, s, nd=3):
    s_txt = f"{s:.{nd}f}" if round(s, nd) > 0 else "{<}" + f"{10 ** -nd:.{nd}f}"
    return f"${m:.{nd}f}\\pm{s_txt}$"


def pmp(x, nd=1):
    x = np.asarray(x, float) * 100
    return f"${np.nanmean(x):.{nd}f}\\pm{np.nanstd(x, ddof=1):.{nd}f}$"


def tup(t, nd=3):
    return pm(t[0], t[1] * F, nd)


def probes_table(d='gait_experiment_output_v2_probes'):
    print('%% ---- tab:probefamilies rows')
    names = {'HHAR (ambulatory)': 'HHAR', 'RealWorld (ambulatory, full)': 'RealWorld, 102.9k',
             'RealWorld (ambulatory, 15.3k)': 'RealWorld, 15.3k',
             'MobiAct B (stairs up/down)': 'MobiAct B (stairs)',
             'MobiAct A (jog/stand, 3.6k)': 'MobiAct A (jog/stand)',
             'MotionSense (ambulatory)': 'MotionSense, ambulatory',
             'MotionSense (all, 5.2k)': 'MotionSense, all'}
    for label, frag, pred, factors in CONDITIONS:
        hits = find_pkl(d, frag, pred)
        res = pickle.load(open(hits[0], 'rb'))
        for fname, (stem, c) in factors.items():
            r = SS.analyse(res, stem, c)
            dag = '$^\\dagger$' if 'MobiAct A' in label else ''
            cells = [pm(np.mean(r['p0']), np.std(r['p0'], ddof=1))]
            for k in ('lin', 'mlp', 'knn', 'best'):
                cells.append(pmp(r['rho_' + k]) + dag)
            cells.append(pm(np.mean(r['Rnl']), np.std(r['Rnl'], ddof=1), 2))
            print(f"{names[label]:24s} & {fname:8s} & " + ' & '.join(cells) + r" \\")
            ls = [f"{x:g}" for x in r['lam_star']]
            print(f"%%   lambda* per seed: {ls}")


def mobiact_r2(d='gait_experiment_output_v2_probes'):
    print('\n%% ---- tab:r2 rows')
    for label, frag, pred in (('jog vs stand', 'results_sweep_mobiact', lambda b: '_cap3579' in b),
                              ('stairs up vs down', 'results_sweep_mobiact', lambda b: '_cap' not in b)):
        res = pickle.load(open(find_pkl(d, frag, pred)[0], 'rb'))
        r = SS.analyse(res, 'probe_activity', 0.5)
        lam1 = res['lambda=1']
        dr = np.array(lam1['disc_act_acc_raw'], float)
        print(label, '| p0', pm(np.mean(r['p0']), np.std(r['p0'], ddof=1)),
              '| lin', pmp(r['rho_lin']), '| best', pmp(r['rho_best']),
              '| disc@1', pm(dr.mean(), dr.std(ddof=1)))


def orig_sweeps(d='gait_experiment_output_v2'):
    print('\n%% ---- original sweeps (stored mean, pop-SD -> sample SD)')
    fn = [p for p in glob.glob(os.path.join(d, 'results_sweep_realworld_dev-waist*seeds0-1-2*.pkl'))
          if '_cap' not in p and 'segdetail' not in p and 'dynamics' not in p][0]
    r = pickle.load(open(fn, 'rb'))
    print('tab:r4 (RealWorld full)', os.path.basename(fn)[-60:])
    for k in sorted(r, key=lambda k: float(k.split('=')[1])):
        v = r[k]
        print(f"{float(k.split('=')[1]):.2f} & {tup(v['probe_activity'])} & {tup(v['probe_device'])} & "
              f"{tup(v['disc_act_acc'])} & {tup(v['disc_dev_acc'])} \\\\")
    for tag, pred in (('full', lambda b: '_cap' not in b), ('15.3k', lambda b: '_cap15315' in b)):
        fn = [p for p in glob.glob(os.path.join(d, 'results_sweep_realworld_dev-waist*seeds0-1-2*.pkl'))
              if pred(os.path.basename(p)) and 'segdetail' not in p][0]
        r = pickle.load(open(fn, 'rb'))
        print(f'\nidentity, RealWorld {tag}: lambda -> in-dist / cross-position / both-unseen (bout acc)')
        for k in sorted(r, key=lambda k: float(k.split('=')[1])):
            c = r[k]['cells']
            print(f"  {k:12s} in-dist {tup(c['seen_seen']['bout_acc'])}  "
                  f"cross-pos {tup(c['unseen_dev_seen_act']['bout_acc'])}")
    fn = [p for p in glob.glob(os.path.join(d, 'results_sweep_hhar_*seeds0-1-2.pkl'))][0]
    r = pickle.load(open(fn, 'rb'))
    print('\nHHAR identity: lambda -> in-dist / cross-device')
    for k in sorted(r, key=lambda k: float(k.split('=')[1])):
        c = r[k]['cells']
        print(f"  {k:12s} in-dist {tup(c['seen_seen']['bout_acc'])}  cross-dev {tup(c['unseen_dev_seen_act']['bout_acc'])}")


if __name__ == '__main__':
    probes_table()
    mobiact_r2()
    orig_sweeps()
