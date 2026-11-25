"""Selection-bias check for the probe-accuracy reduction rho (output of run_selection.ps1).

The paper's rho takes, per seed, the minimum probe accuracy over the lambda sweep on the same held-out cell that is then
reported, which biases rho upward. Here the cell is split into two disjoint halves (by segment, stratified): lambda is
chosen on half A and rho is read on half B, so the reported reduction does not use the data that picked the minimum.

For each condition and factor, mean +/- SD over seeds of
  naive_full   original estimator on the whole cell (minimum over the grid, lambda=0 included), this rerun
  naive_B      the same estimator on half B alone (what selection on the reported data gives at half the data)
  val->test    lambda chosen per seed on half A (minimum of that seed's A curve), rho read on half B at that lambda
  frozen       lambda chosen once from the seed-MEAN A curve (the same lambda for every seed), rho read on B per seed
  lambda=1     rho at the common default lambda=1 on the whole cell (no selection at all)
for the linear probe and the strongest of the three families (lin / MLP / k-NN).
rho = (p0 - p) / (p0 - c), p0 the lambda=0 reading on the same data, c chance; the val->test and frozen estimators can be
negative, naive ones cannot (the minimum includes lambda=0)."""
import glob
import pickle

import numpy as np

D = 'gait_experiment_output_v3_'
CONDS = [
    ('HHAR (ambulatory)', D + 'select_hhar', {'activity': ('probe_activity', 1 / 3), 'device': ('probe_device', 1 / 2)}),
    ('RealWorld (full pool)', D + 'select_realworld_full', {'activity': ('probe_activity', 1 / 3), 'position': ('probe_device', 1 / 6)}),
    ('RealWorld (15.3k)', D + 'select_realworld_cap', {'activity': ('probe_activity', 1 / 3), 'position': ('probe_device', 1 / 6)}),
    ('MobiAct B (stairs)', D + 'select_mobiact_B', {'activity': ('probe_activity', 1 / 2)}),
    ('MobiAct A (jog/stand)', D + 'select_mobiact_A', {'activity': ('probe_activity', 1 / 2)}),
    ('MotionSense (ambulatory)', D + 'select_motionsense_amb', {'activity': ('probe_activity', 1 / 3)}),
    ('MotionSense (all)', D + 'select_motionsense_all', {'activity': ('probe_activity', 1 / 5)}),
]


def load(d):
    f = glob.glob(d + '/*.pkl')
    return pickle.load(open(f[0], 'rb')) if f else None


def curves(r, stem, suffix=''):
    """{family: (seeds, lambdas)} for the whole cell ('') or a half ('_halfA' / '_halfB')"""
    keys = sorted(r, key=lambda k: float(k.split('=')[1]))
    lam = np.array([float(k.split('=')[1]) for k in keys])
    out = {}
    for fs, name in (('', 'lin'), ('_mlp', 'mlp'), ('_knn', 'knn')):
        rows = [r[k].get(f'{stem}{fs}_tc_seen_seen{suffix}_raw', [np.nan]) for k in keys]
        out[name] = np.array(rows, float).T
    return lam, out


def best(P):
    return np.nanmax(np.stack(list(P.values())), axis=0)


def rho(p0, p, c):
    return (p0 - p) / (p0 - c)


def ms(x):
    x = 100 * np.asarray(x, float)
    return f"{np.nanmean(x):6.1f}±{np.nanstd(x, ddof=1) if len(x) > 1 else 0:4.1f}"


def estimators(full, A, B, lam, c, i1):
    """full/A/B: (seeds, lambdas) arrays for one probe definition; returns dict of per-seed rho arrays"""
    n = full.shape[0]
    out = {}
    out['naive_full'] = np.array([rho(full[s, 0], np.nanmin(full[s]), c) for s in range(n)])
    out['naive_B'] = np.array([rho(B[s, 0], np.nanmin(B[s]), c) for s in range(n)])
    istar = np.nanargmin(A, axis=1)
    out['val->test'] = np.array([rho(B[s, 0], B[s, istar[s]], c) for s in range(n)])
    ifrozen = int(np.nanargmin(np.nanmean(A, axis=0)))
    out['frozen'] = np.array([rho(B[s, 0], B[s, ifrozen], c) for s in range(n)])
    out['lambda=1'] = np.array([rho(full[s, 0], full[s, i1], c) for s in range(n)]) if i1 is not None else np.full(n, np.nan)
    out['_istar'] = lam[istar]
    out['_frozen'] = lam[ifrozen]
    return out


rows = []
for label, d, factors in CONDS:
    r = load(d)
    print(f"\n=== {label} ===" + ("" if r else "  (not available)"))
    if not r:
        continue
    for fname, (stem, c) in factors.items():
        lam, Pf = curves(r, stem)
        _, PA = curves(r, stem, '_halfA')
        _, PB = curves(r, stem, '_halfB')
        n = Pf['lin'].shape[0]
        i1 = int(np.where(lam == 1.0)[0][0]) if (lam == 1.0).any() else None
        print(f"  factor {fname}: {n} seeds, chance {c:.3f}, p0(full) lin {np.nanmean(Pf['lin'][:, 0]):.3f} "
              f"best {np.nanmean(best(Pf)[:, 0]):.3f}")
        for fam, (F, A_, B_) in {'linear': (Pf['lin'], PA['lin'], PB['lin']),
                                 'strongest': (best(Pf), best(PA), best(PB))}.items():
            e = estimators(F, A_, B_, lam, c, i1)
            print(f"    {fam:9s} naive_full {ms(e['naive_full'])}  naive_B {ms(e['naive_B'])}  "
                  f"val->test {ms(e['val->test'])}  frozen(lam={e['_frozen']:g}) {ms(e['frozen'])}  lambda=1 {ms(e['lambda=1'])}"
                  f"   lambda* per seed {list(e['_istar'])}")
            rows.append((label, fname, fam, n, e))

print("\n\nRank correlation of p0 (linear, whole cell) with rho over the six conditions of comparable size (descriptive):")
from scipy.stats import spearmanr
six = [('HHAR (ambulatory)', 'activity'), ('RealWorld (15.3k)', 'activity'), ('MobiAct B (stairs)', 'activity'),
       ('MobiAct A (jog/stand)', 'activity'), ('MotionSense (ambulatory)', 'activity'), ('MotionSense (all)', 'activity')]
p0s = []
for label, fname in six:
    d = [c for c in CONDS if c[0] == label][0]
    r = load(d[1])
    if not r:
        p0s = None
        break
    lam, Pf = curves(r, d[2][fname][0])
    p0s.append(float(np.nanmean(Pf['lin'][:, 0])))
if p0s:
    for fam in ('linear', 'strongest'):
        for est in ('naive_full', 'val->test'):
            v = [float(np.nanmean(by_[fam][est])) for by_ in (dict(((l, f), dd) for (l, f), dd in
                 ((k, {fm: e for (lb, fn, fm, nn, e) in rows if (lb, fn) == k}) for k in six)).values())]
            print(f"  {fam:9s} {est:10s} spearman(p0, rho) = {spearmanr(p0s, v).correlation:+.2f}   rho = {[round(100 * x, 1) for x in v]}")

print("\n\nLaTeX rows: condition & factor & linear naive & linear val->test & strongest naive & strongest val->test")
by = {}
for label, fname, fam, n, e in rows:
    by.setdefault((label, fname), {})[fam] = e
for (label, fname), d in by.items():
    f = lambda x: f"${np.nanmean(100 * x):.1f}\\pm{np.nanstd(100 * x, ddof=1) if len(x) > 1 else 0:.1f}$"
    print(f"{label}, {fname} & {f(d['linear']['naive_full'])} & {f(d['linear']['val->test'])} & "
          f"{f(d['strongest']['naive_full'])} & {f(d['strongest']['val->test'])} \\\\")
