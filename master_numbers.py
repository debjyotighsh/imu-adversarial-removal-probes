"""
Single authoritative source for every contested number in the manuscript's HHAR-central /
RealWorld-full / TCN-comparison passages. Reads directly from the raw pkl checkpoints (never
from a previously hand-computed value) and prints everything needed to fix the Sept-2026
consistency critique: the 3-seed vs 5-seed HHAR gap, the RealWorld position balanced accuracy,
the HHAR in-distribution identity trajectory, the HHAR confusion-matrix balanced-acc/macro-F1,
and the 5-seed binomial test against the majority-class rate.

    python master_numbers.py
"""
import pickle

import numpy as np
from scipy import stats

import frozen_analysis as FA
from probe_families_table import find_pkl

HH_FRAG, HH_PRED = 'windows_hz50_win2.56_ov0.5_bike', lambda b: '_cap' not in b and 'dlm5' in b and '_deg' not in b
RW_FRAG, RW_PRED = 'results_sweep_realworld', lambda b: '_cap' not in b


def load(d, frag, pred):
    hits = find_pkl(d, frag, pred)
    assert len(hits) == 1, (d, frag, hits)
    return pickle.load(open(hits[0], 'rb'))


def pm(mean_std, nd=3, pct=False):
    m, s = mean_std
    if pct:
        m, s = 100 * m, 100 * s
    return f"{m:.{nd}f}±{s:.{nd}f}"


print("=" * 78)
print("HHAR central condition: 3-seed (original frozen) vs 5-seed (merged) at lambda=0,1")
print("=" * 78)
hh3 = load('gait_experiment_output_v3_frozen', HH_FRAG, HH_PRED)
hh5 = load('gait_experiment_output_v3_frozen_merged5', HH_FRAG, HH_PRED)

for label, hh in (('3-seed', hh3), ('5-seed', hh5)):
    d0, d1 = hh['lambda=0']['disc_act_acc'], hh['lambda=1']['disc_act_acc']
    p0, p1 = hh['lambda=0']['probe_activity_tc_seen_seen'], hh['lambda=1']['probe_activity_tc_seen_seen']
    gap_pts = 100 * (p1[0] - d1[0])
    ind0 = hh['lambda=0']['cells']['seen_seen']['bout_acc']
    ind1 = hh['lambda=1']['cells']['seen_seen']['bout_acc']
    cross0 = hh['lambda=0']['cells']['unseen_dev_seen_act']['bout_acc']
    cross1 = hh['lambda=1']['cells']['unseen_dev_seen_act']['bout_acc']
    print(f"\n[{label}]")
    print(f"  disc  lambda=0 {pm(d0)}   lambda=1 {pm(d1)}")
    print(f"  probe(held-out) lambda=0 {pm(p0)}   lambda=1 {pm(p1)}")
    print(f"  gap at lambda=1 (probe - disc), points: {gap_pts:.1f}")
    print(f"  in-dist identity  lambda=0 {pm(ind0)}   lambda=1 {pm(ind1)}")
    print(f"  cross-device identity  lambda=0 {pm(cross0)}   lambda=1 {pm(cross1)}")
    print(f"  per-seed disc @ lambda=1: {np.round(hh['lambda=1']['disc_act_acc_raw'], 3)}")
    print(f"  per-seed probe @ lambda=1: {np.round(hh['lambda=1']['probe_activity_tc_seen_seen_raw'], 3)}")

print("\n" + "=" * 78)
print("HHAR lambda=1 discriminator confusion matrix: balanced acc / macro-F1, per seed and mean")
print("=" * 78)
confs = hh5['lambda=1']['disc_act_conf_raw']
bal, f1 = [], []
for s, cf in enumerate(confs):
    acc, b, ent, top = FA.conf_stats(cf)
    macrof1 = FA.conf_stats(cf)
    print(f"  seed {s}: balanced_acc={b:.4f}  entropy={ent:.3f}  top_share={top:.4f}")
    bal.append(b)
print(f"  mean balanced acc (5 seeds) = {np.mean(bal):.4f} +/- {np.std(bal, ddof=1):.4f}")

print("\n" + "=" * 78)
print("RealWorld position, lambda=2: balanced accuracy (3-seed vs 5-seed)")
print("=" * 78)
rw3 = load('gait_experiment_output_v3_frozen', RW_FRAG, RW_PRED)
rw5 = load('gait_experiment_output_v3_frozen_merged5', RW_FRAG, RW_PRED)
for label, rw in (('3-seed', rw3), ('5-seed', rw5)):
    confs = rw['lambda=2'].get('disc_dev_conf_raw')
    if confs is None:
        print(f"  [{label}] no disc_dev_conf_raw stored")
        continue
    bal = []
    for cf in confs:
        acc, b, ent, top = FA.conf_stats(cf)
        bal.append(b)
    print(f"  [{label}] balanced acc per seed: {np.round(bal, 4)}  mean {np.mean(bal):.4f} +/- "
          f"{(np.std(bal, ddof=1) if len(bal) > 1 else 0):.4f}")

print("\n" + "=" * 78)
print("Binomial test: HHAR activity discriminator vs majority-class rate 0.373, lambda=1, ALL 5 SEEDS")
print("=" * 78)
corr = hh5['lambda=1']['disc_act_correct_raw']
n = hh5['lambda=1']['disc_act_n_raw']
for s in range(len(corr)):
    obs = corr[s] / n[s]
    res = stats.binomtest(int(corr[s]), int(n[s]), 0.373, alternative='two-sided')
    print(f"  seed {s}: {corr[s]}/{n[s]} = {obs:.4f} vs chance 0.373, binomial p={res.pvalue:.3g}")

print("\n" + "=" * 78)
print("TCN comparison (from stats_hhar_tcn.log, 3 seeds, for reference -- not recomputed here)")
print("=" * 78)
print("  TCN  lambda=1: disc=0.373  probe(activity)=0.986  gap=61.3 pts")
print(f"  CNN 3-seed lambda=1: disc={hh3['lambda=1']['disc_act_acc'][0]:.3f}  "
      f"probe={hh3['lambda=1']['probe_activity_tc_seen_seen'][0]:.3f}  "
      f"gap={100*(hh3['lambda=1']['probe_activity_tc_seen_seen'][0]-hh3['lambda=1']['disc_act_acc'][0]):.1f} pts")
print(f"  CNN 5-seed lambda=1: disc={hh5['lambda=1']['disc_act_acc'][0]:.3f}  "
      f"probe={hh5['lambda=1']['probe_activity_tc_seen_seen'][0]:.3f}  "
      f"gap={100*(hh5['lambda=1']['probe_activity_tc_seen_seen'][0]-hh5['lambda=1']['disc_act_acc'][0]):.1f} pts")
