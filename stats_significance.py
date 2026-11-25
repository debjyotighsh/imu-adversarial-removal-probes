"""
Statistical significance for the probe/discriminator divergence
==================================================================
Two tests, run against a completed --segment_diagnostics-free checkpoint (any
sweep .pkl from gait_dual_adversarial_v2.py, run after the raw-counts patch —
older checkpoints lack the '*_raw' keys this script reads and will report n/a
for everything).

1. PAIRED TEST ACROSS SEEDS (probe vs. discriminator, same lambda)
   H0: no systematic difference between the two measurements on the same
   representation. Uses the 3 per-seed values directly — the honest test
   given only 3 seeds, not a summary statistic dressed up as one. Reports
   both a paired t-test and the Wilcoxon signed-rank test (the latter needs
   no normality assumption, which matters at n=3); note the Wilcoxon at
   n=3 cannot reach significance at any conventional alpha regardless of
   effect size (the minimum two-sided p-value with 3 pairs is 0.25) — it is
   reported for completeness, not as the basis for any claim here.

2. BINOMIAL TEST AGAINST CHANCE (probe and discriminator, per seed)
   H0: the measured accuracy equals the chance floor. One-sided ('greater')
   for the probe — this is the test that gives the paper's "the information
   is present" claim a p-value instead of just a number well above a dashed
   line. Two-sided for the discriminator — asks whether "converged to
   chance" is itself statistically supportable, or whether the discriminator
   is still (weakly) distinguishable from chance despite reading close to it.
   Each seed's test uses that seed's own correct/n counts (never pooled
   across seeds — the probe's held-out split is IDENTICAL across seeds by
   construction, so the three seeds' test windows are the same windows
   scored by three different encoders, not three independent samples;
   pooling would overstate power).

Usage
-----
    python stats_significance.py --pkl <path/to/results_sweep_..._.pkl> \\
        --lambda 1.0 --factor activity
"""
import argparse
import pickle

import numpy as np
from scipy import stats


def paired_test(vals_a, vals_b, name_a, name_b):
    vals_a, vals_b = np.asarray(vals_a, float), np.asarray(vals_b, float)
    diffs = vals_a - vals_b
    n = len(diffs)
    print(f"\n  {name_a} vs {name_b} across {n} seeds:")
    print(f"    {name_a}: {vals_a.round(4).tolist()}")
    print(f"    {name_b}: {vals_b.round(4).tolist()}")
    print(f"    per-seed difference: {diffs.round(4).tolist()}")
    if n < 2 or np.any(~np.isfinite(diffs)):
        print("    (fewer than 2 finite seeds — no test possible)")
        return
    t_res = stats.ttest_rel(vals_a, vals_b)
    print(f"    paired t-test:      t={t_res.statistic:.3f}, p={t_res.pvalue:.4g}")
    if n >= 3 and len(np.unique(diffs)) > 1:
        w_res = stats.wilcoxon(vals_a, vals_b)
        print(f"    Wilcoxon signed-rank: W={w_res.statistic:.3f}, p={w_res.pvalue:.4g}  "
              f"(floor at n={n} is p=0.25 two-sided — not informative at this n; "
              f"reported for completeness, not as evidence)")
    else:
        print("    Wilcoxon: skipped (all differences identical or n<3)")


def binomial_vs_chance(correct, n, chance, label, alternative):
    if correct is None or n is None or n == 0:
        print(f"    {label}: n/a (missing counts — rerun with the raw-counts patch)")
        return
    acc = correct / n
    res = stats.binomtest(correct, n, chance, alternative=alternative)
    print(f"    {label}: {correct}/{n} = {acc:.4f} (chance {chance:.4f}) "
          f"-> {alternative}, p={res.pvalue:.4g}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pkl', required=True)
    ap.add_argument('--lambda', dest='lam', type=float, required=True)
    ap.add_argument('--factor', choices=['activity', 'device'], default='activity')
    ap.add_argument('--chance', type=float, default=None,
                    help="Chance floor for the PROBE binomial test (balanced "
                         "accuracy's chance level, i.e. 1/number of seen classes). "
                         "Required for test [2]; the script cannot recover the "
                         "class count from the checkpoint alone.")
    ap.add_argument('--disc_null', type=float, default=None,
                    help="Null accuracy for the DISCRIMINATOR binomial test. The "
                         "discriminator's accuracy is a raw training accuracy, so "
                         "under class imbalance an uninformative classifier that "
                         "always predicts the majority class scores the majority-"
                         "class share, NOT 1/K. Pass that share here (HHAR "
                         "ambulatory training pool: 0.3726 vs. uniform 0.3333). "
                         "Defaults to --chance, which is only correct if classes "
                         "are balanced.")
    args = ap.parse_args()

    with open(args.pkl, 'rb') as fh:
        results = pickle.load(fh)
    key = f"lambda={args.lam:g}"
    if key not in results:
        raise SystemExit(f"'{key}' not in checkpoint. Have: {sorted(results)}")
    r = results[key]

    probe_k, disc_k = f'probe_{args.factor}', f'disc_{args.factor[:3]}_acc'
    print(f"{'='*70}\n{args.pkl}\n{key}  factor={args.factor}\n{'='*70}")

    print("\n[1] PAIRED TEST ACROSS SEEDS — probe vs. discriminator, same lambda")
    probe_raw = r.get(f'{probe_k}_raw')
    disc_raw = r.get(f'{disc_k}_raw')
    if probe_raw is None or disc_raw is None:
        print("  n/a — this checkpoint predates the raw-per-seed patch. Rerun to get it.")
    else:
        paired_test(probe_raw, disc_raw, probe_k, disc_k)

    print("\n[2] BINOMIAL TEST AGAINST CHANCE — per seed, not pooled")
    chance = args.chance
    if chance is None:
        print("  (no --chance given: skipping — pass the true chance floor, e.g. "
              "--chance 0.333 for a 3-class factor)")
    else:
        pc_raw = r.get(f'{probe_k}_correct_raw')
        pn_raw = r.get(f'{probe_k}_n_raw')
        dc_raw = r.get(f'{disc_k[:-4]}_correct_raw')   # disc_act_acc -> disc_act
        dn_raw = r.get(f'{disc_k[:-4]}_n_raw')
        if pc_raw is None or dc_raw is None:
            print("  n/a — this checkpoint predates the raw-counts patch. Rerun to get it.")
        else:
            for i in range(len(pc_raw)):
                print(f"\n  seed index {i}:")
                binomial_vs_chance(pc_raw[i], pn_raw[i], chance,
                                   f'probe_{args.factor} (H1: > chance)', 'greater')
                null = args.disc_null if args.disc_null is not None else chance
                binomial_vs_chance(dc_raw[i], dn_raw[i], null,
                                   f'disc_{args.factor} (H1: != {null:.4f})', 'two-sided')


if __name__ == '__main__':
    main()
