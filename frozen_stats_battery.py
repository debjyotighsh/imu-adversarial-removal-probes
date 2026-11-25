"""
Statistical test battery requested by the reviewer table:

    Baseline vs adversarial              paired Wilcoxon
    Linear vs MLP probe                  paired Wilcoxon
    Probe accuracy across lambda         mixed-effects model / repeated-measures test
    Discriminator vs chance              binomial test
    Multiple lambda pairwise tests       Holm correction

Every test is run on the frozen rerun (gait_experiment_output_v3_frozen), the held-out probe
where a probe is involved. n=3 seeds throughout, so every result states what power that gives
before reporting a p-value.

    python frozen_stats_battery.py
"""
import itertools
import pickle

import numpy as np
from scipy import stats
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests

from probe_families_table import CONDITIONS, find_pkl

D = 'gait_experiment_output_v3_frozen'


def load(frag, pred):
    hits = find_pkl(D, frag, pred)
    return pickle.load(open(hits[0], 'rb')) if hits else None


def wilcoxon_note(x, y, label):
    """paired Wilcoxon signed-rank; states the n=3 power ceiling explicitly."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    diff = x - y
    n = len(diff)
    try:
        res = stats.wilcoxon(x, y, zero_method='wilcox')
        p = res.pvalue
    except ValueError as e:                       # all-zero differences, etc.
        p = float('nan')
        print(f"  [{label}] n={n}: Wilcoxon failed ({e})")
        return
    min_p = 1 / 2 ** (n - 1) if n <= 20 else None  # exact two-sided floor for small n, no ties
    pstr = f"{p:.3g}" if p < 1e-3 else f"{p:.3f}"
    print(f"  [{label}] n={n} pairs, mean diff {diff.mean():+.4f} (x-y): "
          f"Wilcoxon p={pstr}" + (f"  (exact floor at n={n} is {min_p:.3f})" if min_p else ""))


# =====================================================================
print("=" * 70)
print("1. BASELINE (lambda=0) vs ADVERSARIAL (lambda=1), paired by seed, n=3")
print("=" * 70)
hh = load('windows_hz50_win2.56_ov0.5_bike', lambda b: '_cap' not in b and 'dlm5' in b and '_deg' not in b)
for metric, key in (('discriminator acc', 'disc_act_acc_raw'), ('held-out probe', 'probe_activity_tc_seen_seen_raw')):
    wilcoxon_note(hh['lambda=0'][key], hh['lambda=1'][key], f'HHAR activity, {metric}')

rw = load('results_sweep_realworld', lambda b: '_cap' not in b)
for metric, key in (('discriminator acc', 'disc_act_acc_raw'), ('held-out probe', 'probe_activity_tc_seen_seen_raw')):
    wilcoxon_note(rw['lambda=0'][key], rw['lambda=0.15'][key], f'RealWorld activity, {metric} (0 vs 0.15, the dip)')
    wilcoxon_note(rw['lambda=0'][key], rw['lambda=1'][key], f'RealWorld activity, {metric} (0 vs 1)')

# =====================================================================
print()
print("=" * 70)
print("2. LINEAR vs MLP PROBE, held-out, paired WITHIN (lambda, seed) cells")
print("=" * 70)


def lin_vs_mlp(r, stem, label):
    lams = sorted(r, key=lambda k: float(k.split('=')[1]))
    lin, mlp = [], []
    for k in lams:
        l_ = r[k].get(f'{stem}_tc_seen_seen_raw')
        m_ = r[k].get(f'{stem}_mlp_tc_seen_seen_raw')
        if l_ is None or m_ is None:
            continue
        lin.extend(l_)
        mlp.extend(m_)
    wilcoxon_note(mlp, lin, f'{label} (MLP - linear, all lambda x seed cells)')


lin_vs_mlp(hh, 'probe_activity', 'HHAR activity')
lin_vs_mlp(rw, 'probe_activity', 'RealWorld activity (full pool)')
lin_vs_mlp(rw, 'probe_device', 'RealWorld position (full pool)')
mb = load('results_sweep_mobiact', lambda b: '_cap' not in b)
lin_vs_mlp(mb, 'probe_activity', 'MobiAct stairs')

# =====================================================================
print()
print("=" * 70)
print("3. PROBE ACCURACY ACROSS LAMBDA: repeated-measures (Friedman) and mixed-effects (MixedLM)")
print("=" * 70)


def across_lambda(r, stem, label):
    lams = sorted((k for k in r if r[k].get(f'{stem}_tc_seen_seen_raw') is not None),
                  key=lambda k: float(k.split('=')[1]))
    mat = np.array([r[k][f'{stem}_tc_seen_seen_raw'] for k in lams])   # (n_lambda, n_seed)
    # Friedman: nonparametric repeated-measures across lambda, subject = seed
    fr = stats.friedmanchisquare(*mat.tolist())
    print(f"\n  {label}: {len(lams)} lambda points x 3 seeds")
    print(f"    Friedman (nonparametric repeated-measures, subject=seed): "
          f"chi2={fr.statistic:.2f}, df={len(lams)-1}, p={fr.pvalue:.4f}")
    # Mixed-effects: probe ~ lambda (categorical fixed effect) + (1|seed)
    import pandas as pd
    rows = []
    for li, k in enumerate(lams):
        lam = float(k.split('=')[1])
        for s, v in enumerate(r[k][f'{stem}_tc_seen_seen_raw']):
            rows.append(dict(probe=v, lam=str(lam), seed=s))
    df = pd.DataFrame(rows)
    try:
        md = MixedLM.from_formula('probe ~ C(lam)', groups='seed', data=df)
        fit = md.fit(reml=False)
        # likelihood-ratio test of the lambda effect against an intercept-only null
        md0 = MixedLM.from_formula('probe ~ 1', groups='seed', data=df)
        fit0 = md0.fit(reml=False)
        lr = 2 * (fit.llf - fit0.llf)
        dfree = len(lams) - 1
        p_lr = stats.chi2.sf(lr, dfree)
        print(f"    Mixed-effects (probe ~ C(lambda) + (1|seed)), LR test vs intercept-only: "
              f"LR={lr:.2f}, df={dfree}, p={p_lr:.4f}")
    except Exception as e:                        # noqa: BLE001
        print(f"    Mixed-effects model failed: {type(e).__name__}: {e}")


across_lambda(hh, 'probe_activity', 'HHAR activity')
across_lambda(rw, 'probe_activity', 'RealWorld activity (full pool)')
across_lambda(rw, 'probe_device', 'RealWorld position (full pool)')

# =====================================================================
print()
print("=" * 70)
print("4. DISCRIMINATOR vs CHANCE, two-sided binomial test on training windows")
print("=" * 70)


def disc_vs_chance(r, lam_key, correct_key, n_key, chance, label):
    corr = r[lam_key].get(correct_key)
    n = r[lam_key].get(n_key)
    if corr is None:
        print(f"  [{label}] no correct/n counts stored")
        return
    for s in range(len(corr)):
        res = stats.binomtest(int(corr[s]), int(n[s]), chance, alternative='two-sided')
        print(f"  [{label}] seed {s}: {corr[s]}/{n[s]} = {corr[s]/n[s]:.4f} vs chance {chance:.4f}, "
              f"binomial p={res.pvalue:.3g}")


disc_vs_chance(hh, 'lambda=1', 'disc_act_correct_raw', 'disc_act_n_raw', 0.373, 'HHAR activity vs majority-class rate, lambda=1')
disc_vs_chance(rw, 'lambda=2', 'disc_dev_correct_raw', 'disc_dev_n_raw', 1 / 6, 'RealWorld position vs uniform chance, lambda=2')
disc_vs_chance(mb, 'lambda=1', 'disc_act_correct_raw', 'disc_act_n_raw', 0.5, 'MobiAct stairs vs chance, lambda=1')

# =====================================================================
print()
print("=" * 70)
print("5. PAIRWISE LAMBDA COMPARISONS, paired Wilcoxon (n=3) + Holm correction")
print("=" * 70)


def pairwise_holm(r, stem, label):
    lams = sorted((k for k in r if r[k].get(f'{stem}_tc_seen_seen_raw') is not None),
                  key=lambda k: float(k.split('=')[1]))
    pairs, pvals = [], []
    for a, b in itertools.combinations(lams, 2):
        xa = r[a][f'{stem}_tc_seen_seen_raw']
        xb = r[b][f'{stem}_tc_seen_seen_raw']
        try:
            p = stats.wilcoxon(xa, xb, zero_method='wilcox').pvalue
        except ValueError:
            p = 1.0
        pairs.append((a, b))
        pvals.append(p)
    rej, p_adj, _, _ = multipletests(pvals, alpha=0.05, method='holm')
    n_sig = int(rej.sum())
    print(f"\n  {label}: {len(pairs)} pairwise lambda comparisons (n=3 each), Holm-corrected at alpha=0.05")
    print(f"    raw p-values range {min(pvals):.3f}-{max(pvals):.3f}; "
          f"{n_sig} of {len(pairs)} remain significant after Holm correction")
    for (a, b), p, padj, r_ in sorted(zip(pairs, pvals, p_adj, rej), key=lambda t: t[2])[:5]:
        print(f"      {a} vs {b}: raw p={p:.3f}, Holm-adjusted p={padj:.3f}, significant={r_}")


pairwise_holm(hh, 'probe_activity', 'HHAR activity')
pairwise_holm(rw, 'probe_activity', 'RealWorld activity (full pool)')
