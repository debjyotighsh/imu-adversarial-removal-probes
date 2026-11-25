"""
Analysis of the frozen rerun (gait_experiment_output_v3_frozen)
================================================================
1. Probes scored on windows the encoder never saw (test cells) against the in-pool probe.
2. What the discriminator predicts at the end of training: accuracy, balanced accuracy,
   predicted-class entropy and the share of the most-predicted class. A constant predictor
   has entropy 0; a discriminator at chance with spread-out predictions has entropy near 1.
3. The removal statistics (rho) computed on the test-cell probes.

    python frozen_analysis.py --dir gait_experiment_output_v3_frozen
"""
import argparse
import glob
import os
import pickle

import numpy as np

import seed_stats as SS
from probe_families_table import CONDITIONS, find_pkl

F = np.sqrt(1.5)


def ms(x, nd=3):
    x = np.asarray(x, float)
    return f"{np.nanmean(x):.{nd}f}±{np.nanstd(x, ddof=1):.{nd}f}"


def conf_stats(conf):
    """conf: (K, K) counts, rows = true class. Returns accuracy, balanced acc, entropy, top-share."""
    conf = np.asarray(conf, float)
    n = conf.sum()
    acc = np.trace(conf) / n
    recall = np.diag(conf) / np.maximum(conf.sum(1), 1)
    q = conf.sum(0) / n
    nz = q[q > 0]
    ent = float(-(nz * np.log(nz)).sum() / np.log(len(q)))
    return acc, float(recall.mean()), ent, float(q.max())


def load(d, frag, pred):
    hits = find_pkl(d, frag, pred)
    return pickle.load(open(hits[0], 'rb')) if hits else None


def part1(d):
    print('\n=== 1. probes on windows the encoder never saw (mean±SD over seeds) ===')
    for label, frag, pred, factors in CONDITIONS:
        r = load(d, frag, pred)
        if r is None:
            continue
        for fname, (stem, c) in factors.items():
            lam = sorted(float(k.split('=')[1]) for k in r)
            print(f"\n{label} -- {fname} (chance {c:.3f})")
            print(f"{'lambda':>7} {'in-pool lin':>13} {'held-out cell lin':>18} {'mlp':>13} {'knn':>13} {'best':>13}")
            _, P_in = SS.seed_curves(r, stem)
            _, P_tc = SS.seed_curves(r, stem, 'seen_seen')
            if 'lin' not in P_tc or not np.isfinite(P_tc['lin']).any():
                print('   (no held-out cell for this factor)')
                continue
            best = np.nanmax(np.stack(list(P_tc.values())), axis=0)
            for j, l in enumerate(lam):
                print(f"{l:7g} {ms(P_in['lin'][:, j]):>13} {ms(P_tc['lin'][:, j]):>18} "
                      f"{ms(P_tc.get('mlp', np.full((1, len(lam)), np.nan))[:, j]):>13} "
                      f"{ms(P_tc.get('knn', np.full((1, len(lam)), np.nan))[:, j]):>13} {ms(best[:, j]):>13}")
            for tcn in ('unseen_dev_seen_act', 'seen_dev_unseen_act'):
                _, P2 = SS.seed_curves(r, stem, tcn)
                if 'lin' in P2 and np.isfinite(P2['lin']).any():
                    j = len(lam) - 1
                    print(f"   shifted cell '{tcn}': lin at lambda={lam[j]:g} {ms(P2['lin'][:, j])}, at 0 {ms(P2['lin'][:, 0])}")


def part2(d):
    print('\n=== 2. what the discriminator predicts (final epoch, mean over seeds) ===')
    print(f"{'condition':<30}{'factor':<10}{'lambda':>7}{'acc':>8}{'bal.acc':>9}{'chance':>8}{'entropy':>9}{'top-share':>10}")
    for label, frag, pred, factors in CONDITIONS:
        r = load(d, frag, pred)
        if r is None:
            continue
        for fname, key, c in (('activity', 'disc_act_conf_raw', None), ('device', 'disc_dev_conf_raw', None)):
            if fname == 'device' and 'device' not in factors and 'position' not in factors:
                continue
            for k in sorted(r, key=lambda k: float(k.split('=')[1])):
                confs = r[k].get(key)
                if not confs or confs[0] is None:
                    continue
                st = np.array([conf_stats(cf) for cf in confs])
                K = np.asarray(confs[0]).shape[0]
                lam = float(k.split('=')[1])
                if lam in (0.0, 0.1, 1.0, 2.0) or k == max(r, key=lambda x: float(x.split('=')[1])):
                    print(f"{label:<30}{('position' if 'realworld' in frag and fname == 'device' else fname):<10}{lam:7g}"
                          f"{st[:, 0].mean():8.3f}{st[:, 1].mean():9.3f}{1 / K:8.3f}{st[:, 2].mean():9.3f}{st[:, 3].mean():10.3f}")


def part3(d):
    print('\n=== 3. rho on in-pool vs held-out-cell probes (mean±SD %) ===')
    print(f"{'condition':<32}{'factor':<10}{'rho_lin in-pool':>17}{'rho_lin cell':>15}{'rho_best in-pool':>18}{'rho_best cell':>15}")
    for label, frag, pred, factors in CONDITIONS:
        r = load(d, frag, pred)
        if r is None:
            continue
        for fname, (stem, c) in factors.items():
            try:
                a = SS.analyse(r, stem, c)
                b = SS.analyse(r, stem, c, tc='seen_seen')
            except Exception as e:                   # noqa: BLE001
                print(f"{label:<32}{fname:<10} {type(e).__name__}: {e}")
                continue
            print(f"{label:<32}{fname:<10}{ms(100*a['rho_lin'], 1):>17}{ms(100*b['rho_lin'], 1):>15}"
                  f"{ms(100*a['rho_best'], 1):>18}{ms(100*b['rho_best'], 1):>15}")


def part4(d, tol=0.05, ent_min=0.9):
    """Every (condition, factor, lambda, seed) where the discriminator is within `tol` of chance in
    BALANCED accuracy and its predictions are spread over the classes (normalised predicted-class
    entropy >= ent_min), together with the linear probe on the held-out cell for that same run."""
    print(f"\n=== 4. discriminator at chance WITH spread-out predictions (bal.acc within {tol} of chance, "
          f"entropy >= {ent_min}) ===")
    found = 0
    total = 0
    for label, frag, pred, factors in CONDITIONS:
        r = load(d, frag, pred)
        if r is None:
            continue
        for fname, key, stem in (('activity', 'disc_act_conf_raw', 'probe_activity'),
                                 ('device/position', 'disc_dev_conf_raw', 'probe_device')):
            if fname != 'activity' and not any(k in factors for k in ('device', 'position')):
                continue
            for k in sorted(r, key=lambda k: float(k.split('=')[1])):
                lam = float(k.split('=')[1])
                if lam == 0:
                    continue
                confs = r[k].get(key)
                if not confs or confs[0] is None:
                    continue
                K = np.asarray(confs[0]).shape[0]
                pin = r[k].get(stem + '_raw')
                ptc = r[k].get(stem + '_tc_seen_seen_raw')
                for s, cf in enumerate(confs):
                    acc, bal, ent, top = conf_stats(cf)
                    total += 1
                    if bal - 1 / K < tol and ent >= ent_min:
                        found += 1
                        print(f"  {label:<30}{fname:<16}lambda={lam:<5g} seed {s}: bal.acc {bal:.3f} (chance {1/K:.3f}) "
                              f"entropy {ent:.2f} top-share {top:.2f} | probe in-pool {pin[s] if pin else float('nan'):.3f}"
                              f" held-out {ptc[s] if ptc else float('nan'):.3f}")
    print(f"  {found} of {total} discriminator runs (lambda > 0) meet the criterion")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='gait_experiment_output_v3_frozen')
    a = ap.parse_args()
    part1(a.dir)
    part2(a.dir)
    part3(a.dir)
    part4(a.dir)
