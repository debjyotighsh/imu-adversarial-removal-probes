"""Compare the intervention runs with the matched baseline (seeds 0-2 of run_disc_heldout.ps1's output)."""
import glob
import pickle

import numpy as np

import frozen_analysis as FA

RUNS = [('baseline', 'gait_experiment_output_v3_disc_heldout'),
        ('K=5 disc steps', 'gait_experiment_output_v3_intervene_k5'),
        ('encoder lr x0.1', 'gait_experiment_output_v3_intervene_bb0p1'),
        ('encoder lr x0.03', 'gait_experiment_output_v3_intervene_bb0p03'),
        ('lr x0.1 + K=5', 'gait_experiment_output_v3_intervene_bb0p1_k5')]


def pm(v, nd=3):
    v = np.asarray(v, float)
    return f"{v.mean():.{nd}f}±{(v.std(ddof=1) if len(v) > 1 else 0):.{nd}f}"


for lam in ('lambda=0.1', 'lambda=1'):
    print(f"\n--- {lam} (first three seeds) ---")
    print(f"{'condition':<18} {'disc train bal':>15} {'disc held-out':>14} {'entropy':>8} {'lin probe':>10} "
          f"{'MLP':>7} {'kNN':>7} {'gap(lin)':>9} {'gap(MLP)':>9} {'id win-acc seen/unseen-dev':>30}")
    for name, d in RUNS:
        f = glob.glob(d + '/*.pkl')
        if not f:
            print(f"{name:<18} (not finished)")
            continue
        v = pickle.load(open(f[0], 'rb'))[lam]
        n = 3
        held = np.asarray(v['disc_act_heldout_tc_seen_seen_raw'])[:n]
        lin = np.asarray(v['probe_activity_tc_seen_seen_raw'])[:n]
        mlp = np.asarray(v['probe_activity_mlp_tc_seen_seen_raw'])[:n]
        knn = np.asarray(v['probe_activity_knn_tc_seen_seen_raw'])[:n]
        st = [FA.conf_stats(cf) for cf in v['disc_act_conf_raw'][:n]]
        trb = [s[1] for s in st]
        ent = np.mean([s[2] for s in st])
        # identity (the target task) window accuracy on the held-out cells
        # stored as (mean, SD) over all seeds in the run; the baseline run has five seeds, the others three
        idv = (f"{v['cells']['seen_seen']['window_acc'][0]:.3f} / "
               f"{v['cells']['unseen_dev_seen_act']['window_acc'][0]:.3f}")
        print(f"{name:<18} {pm(trb):>15} {pm(held):>14} {ent:8.2f} {lin.mean():10.3f} {mlp.mean():7.3f} {knn.mean():7.3f} "
              f"{100*(lin.mean()-held.mean()):9.1f} {100*(mlp.mean()-held.mean()):9.1f} {idv:>30}")
