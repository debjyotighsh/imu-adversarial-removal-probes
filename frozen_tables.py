"""
LaTeX-ready rows for every table, from the frozen rerun (gait_experiment_output_v3_frozen).
Everything is mean +/- SD over three seeds (sample SD, n-1). rho is computed per seed.

    python frozen_tables.py --dir gait_experiment_output_v3_frozen [--cell]

--cell uses the probes scored on the held-out evaluation cell (seen_seen) instead of the
in-pool probes.
"""
import argparse
import glob
import os
import pickle

import numpy as np

import seed_stats as SS
from probe_families_table import CONDITIONS, find_pkl

F = np.sqrt(1.5)


def pm(m, s, nd=3):
    s_txt = f"{s:.{nd}f}" if round(s, nd) > 0 else "{<}" + f"{10 ** -nd:.{nd}f}"
    return f"${m:.{nd}f}\\pm{s_txt}$"


def pmv(x, nd=3, scale=1.0):
    x = np.asarray(x, float) * scale
    return pm(np.nanmean(x), np.nanstd(x, ddof=1), nd)


def tup(t, nd=3):
    return pm(t[0], t[1] * F, nd)


def pick(d, frag, pred):
    hits = find_pkl(d, frag, pred)
    return pickle.load(open(hits[0], 'rb')) if hits else None


def raw(r, key):
    return np.array(r[key + '_raw'], float)


def table_hhar(d, tag='dlm5'):
    r = pick(d, 'windows_hz50_win2.56_ov0.5_bike', lambda b: '_cap' not in b and tag in b)
    print(f"\n%% ---- Table (HHAR, {tag}): lambda | disc | held-out-cell probe | in-pool probe | identity cross-dev | in-dist")
    for k in sorted(r, key=lambda k: float(k.split('=')[1])):
        v = r[k]
        lam = float(k.split('=')[1])
        cell = v.get('probe_activity_tc_seen_seen_raw')
        print(f"{lam:.2f} & {pmv(v['disc_act_acc_raw'])} & "
              f"{pmv(cell) if cell is not None else 'n/a'} & {pmv(v['probe_activity_raw'])} & "
              f"{tup(v['cells']['unseen_dev_seen_act']['bout_acc'])} & {tup(v['cells']['seen_seen']['bout_acc'])} \\\\")


def table_families(d, tc=None):
    print(f"\n%% ---- probe families ({'held-out cell ' + tc if tc else 'in-pool'})")
    names = {'HHAR (ambulatory)': 'HHAR', 'RealWorld (ambulatory, full)': 'RealWorld, 102.9k',
             'RealWorld (ambulatory, 15.3k)': 'RealWorld, 15.3k',
             'MobiAct B (stairs up/down)': 'MobiAct B (stairs)',
             'MobiAct A (jog/stand, 3.6k)': 'MobiAct A (jog/stand)',
             'MotionSense (ambulatory)': 'MotionSense, ambulatory',
             'MotionSense (all, 5.2k)': 'MotionSense, all'}
    for label, frag, pred, factors in CONDITIONS:
        r = pick(d, frag, pred)
        if r is None:
            continue
        for fname, (stem, c) in factors.items():
            a = SS.analyse(r, stem, c, tc=tc)
            if not np.isfinite(a['p0']).any():
                continue
            dag = '$^\\dagger$' if 'MobiAct A' in label else ''
            cells = [pmv(a['p0'])] + [pmv(a['rho_' + k], 1, 100) + dag for k in ('lin', 'mlp', 'knn', 'best')]
            cells.append(pmv(a['Rnl'], 2))
            print(f"{names.get(label, label):24s} & {fname:8s} & " + ' & '.join(cells) +
                  f" & {', '.join(f'{x:g}' for x in a['lam_star'])} \\\\")


def table_realworld(d):
    r = pick(d, 'results_sweep_realworld', lambda b: '_cap' not in b)
    if r is None:
        print('\n%% RealWorld full run not present yet')
        return
    print("\n%% ---- RealWorld full: lambda | probe_act (cell) | probe_pos (cell) | disc_act | disc_pos | probe_act (in-pool) | probe_pos (in-pool)")
    for k in sorted(r, key=lambda k: float(k.split('=')[1])):
        v = r[k]
        lam = float(k.split('=')[1])
        g = lambda key: pmv(v[key + '_raw']) if (key + '_raw') in v else 'n/a'
        print(f"{lam:.2f} & {g('probe_activity_tc_seen_seen')} & {g('probe_device_tc_seen_seen')} & "
              f"{g('disc_act_acc')} & {g('disc_dev_acc')} & {g('probe_activity')} & {g('probe_device')} \\\\")
    for tag, pred in (('full', lambda b: '_cap' not in b), ('15.3k', lambda b: '_cap15315' in b)):
        rr = pick(d, 'results_sweep_realworld', pred)
        if rr is None:
            continue
        print(f"\n%% identity, RealWorld {tag}: lambda | in-dist | cross-position")
        for k in sorted(rr, key=lambda k: float(k.split('=')[1])):
            c = rr[k]['cells']
            print(f"  {k:12s} {tup(c['seen_seen']['bout_acc'])}  {tup(c['unseen_dev_seen_act']['bout_acc'])}")


def table_mobiact(d):
    print("\n%% ---- MobiAct rows: p0 | rho_lin | rho_best | disc@lambda=1")
    for label, pred in (('jog vs stand', lambda b: '_cap3579' in b), ('stairs up vs down', lambda b: '_cap' not in b)):
        r = pick(d, 'results_sweep_mobiact', pred)
        if r is None:
            continue
        a = SS.analyse(r, 'probe_activity', 0.5)
        print(label, '|', pmv(a['p0']), '|', pmv(a['rho_lin'], 1, 100), '|', pmv(a['rho_best'], 1, 100),
              '|', pmv(r['lambda=1']['disc_act_acc_raw']))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='gait_experiment_output_v3_frozen')
    a = ap.parse_args()
    table_hhar(a.dir, 'dlm5')
    table_hhar(a.dir, 'dlm1')
    table_families(a.dir)
    table_families(a.dir, tc='seen_seen')
    table_mobiact(a.dir)
    table_realworld(a.dir)
