"""
Linear removability vs. nonlinear recoverability
=================================================
Reads the sweeps written with --nonlinear_probe --knn_probe and, per condition
and per factor, reports how much of the factor each probe family still recovers.

Definitions (chance c is the factor's uniform chance level):

  rho_probe   = (p0 - p_min) / (p0 - c)     removal as seen by ONE probe family,
                p0 = balanced accuracy at lambda=0, p_min = its minimum over the sweep.
  rho_lin     the paper's Eq. (1): rho_probe for the linear probe.
                "Linear removability": how much a LINEAR classifier loses.
  rho_best    rho_probe computed on the per-lambda MAXIMUM over probe families,
                p_best(lambda) = max(linear, MLP, kNN). Every probe's accuracy is a
                lower bound on what is recoverable, so the maximum is the tightest
                bound available; rho_best is removal as seen by the strongest probe.
  R_nl        (p_nl(lambda*) - c) / (p_nl(0) - c), p_nl = max(MLP, kNN), with
                lambda* the lambda at which the LINEAR probe is lowest.
                "Nonlinear recoverability": the share of baseline decodability a
                nonlinear probe still recovers where the linear probe is lowest.

Asymmetry: a high probe accuracy is evidence the factor remains decodable; a low
LINEAR accuracy is evidence only of linear removal.

    python probe_families_table.py --dir gait_experiment_output_v2_probes
"""
import argparse
import glob
import os
import pickle

import numpy as np

# (label, filename fragment, predicate on the basename, {factor: (key stem, chance)})
CONDITIONS = [
    ('HHAR (ambulatory)', 'windows_hz50_win2.56_ov0.5_bike',
     lambda b: '_cap' not in b and 'dlm5' in b,
     {'activity': ('probe_activity', 1 / 3), 'device': ('probe_device', 1 / 2)}),
    ('RealWorld (ambulatory, full)', 'results_sweep_realworld',
     lambda b: '_cap' not in b,
     {'activity': ('probe_activity', 1 / 3), 'position': ('probe_device', 1 / 6)}),
    ('RealWorld (ambulatory, 15.3k)', 'results_sweep_realworld',
     lambda b: '_cap15315' in b,
     {'activity': ('probe_activity', 1 / 3), 'position': ('probe_device', 1 / 6)}),
    ('MobiAct B (stairs up/down)', 'results_sweep_mobiact',
     lambda b: '_cap' not in b,
     {'activity': ('probe_activity', 1 / 2)}),
    ('MobiAct A (jog/stand, 3.6k)', 'results_sweep_mobiact',
     lambda b: '_cap3579' in b,
     {'activity': ('probe_activity', 1 / 2)}),
    ('MotionSense (ambulatory)', 'results_sweep_motionsense',
     lambda b: '_cap' not in b,
     {'activity': ('probe_activity', 1 / 3)}),
    ('MotionSense (all, 5.2k)', 'results_sweep_motionsense',
     lambda b: '_cap5234' in b,
     {'activity': ('probe_activity', 1 / 5)}),
]


def find_pkl(d, frag, pred):
    return [p for p in sorted(glob.glob(os.path.join(d, '*.pkl')))
            if frag in os.path.basename(p)
            and 'segdetail' not in p and 'dynamics' not in p and '_deg' not in os.path.basename(p)
            and pred(os.path.basename(p))]


def curve(results, stem):
    keys = sorted(results, key=lambda k: float(k.split('=')[1]))
    lam = np.array([float(k.split('=')[1]) for k in keys])
    out = {}
    for suffix, name in (('', 'lin'), ('_mlp', 'mlp'), ('_knn', 'knn')):
        out[name] = np.array([results[k].get(stem + suffix, (np.nan, np.nan))[0] for k in keys])
    return lam, out


def rho(p, c):
    if not np.isfinite(p[0]) or p[0] - c <= 1e-6:
        return np.nan
    return (p[0] - np.nanmin(p)) / (p[0] - c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='gait_experiment_output_v2_probes')
    ap.add_argument('--per_lambda', action='store_true', help="also print every lambda row")
    args = ap.parse_args()

    for label, frag, pred, factors in CONDITIONS:
        hits = find_pkl(args.dir, frag, pred)
        if not hits:
            print(f"\n{label}: (no checkpoint found)")
            continue
        results = pickle.load(open(hits[0], 'rb'))
        for fname, (stem, c) in factors.items():
            lam, P = curve(results, stem)
            if not np.isfinite(P['lin']).any():
                continue
            fam = {k: v for k, v in P.items() if np.isfinite(v).any()}
            best = np.nanmax(np.vstack([fam[k] for k in fam]), axis=0)
            i_star = int(np.nanargmin(P['lin']))
            others = [k for k in fam if k != 'lin']
            nl = np.nanmax(np.vstack([fam[k] for k in others]), axis=0) if others else None
            print(f"\n{label} -- {fname}  (chance {c:.3f}; {len(lam)} lambdas)")
            print(f"  {'':>10}" + "".join(f"{k:>8}" for k in fam))
            print(f"  {'p0':>10}" + "".join(f"{fam[k][0]:8.3f}" for k in fam))
            print(f"  {'p_min':>10}" + "".join(f"{np.nanmin(fam[k]):8.3f}" for k in fam))
            print(f"  {'@lambda*':>10}" + "".join(f"{fam[k][i_star]:8.3f}" for k in fam)
                  + f"    (lambda* = {lam[i_star]:g}, where the linear probe is lowest)")
            print(f"  {'rho':>10}" + "".join(f"{rho(fam[k], c):8.3f}" for k in fam))
            print(f"  rho_best (max over families per lambda) = {rho(best, c):.3f}")
            if nl is not None and nl[0] - c > 1e-6:
                print(f"  R_nl (nonlinear share still recovered at lambda*) = "
                      f"{(nl[i_star] - c) / (nl[0] - c):.3f}")
            if args.per_lambda:
                print("  lambda " + "".join(f"{k:>8}" for k in fam) + "    best")
                for j, l in enumerate(lam):
                    print(f"  {l:6g} " + "".join(f"{fam[k][j]:8.3f}" for k in fam)
                          + f"  {best[j]:8.3f}")


if __name__ == '__main__':
    main()

