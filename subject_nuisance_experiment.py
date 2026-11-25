"""
Inverted-configuration validation: activity is the TARGET, subject identity is the NUISANCE
removed by adversarial training. This is the configuration used by the generalisation papers
in the audit (Table 1) -- Suh et al., Han et al., Calatrava-Nicolas et al. -- as opposed to the
rest of this paper, where identity is the target and activity/device/position are nuisances.

Reuses GaitModel/train_model/evaluate_everything unchanged, by feeding swapped label arrays:
  y_subj (normally identity, always a plain task head)      <- activity labels
  y_act  (normally activity, GRL-adversarial nuisance head) <- subject labels
  n_subj <- number of activities;  n_act <- number of TRAINING subjects
Held-out subjects are excluded from training entirely (label -1 in the subject/nuisance slot,
same convention _run_experiments() uses for a held-out device or activity) and evaluated as a
separate cell: does activity recognition generalise to unseen subjects as lambda rises. The
post-hoc probe (Section 3.6's methodology, unchanged) is fit on the training pool and scored on
held-out segments from the SAME (seen) subjects, exactly as every other condition in this paper
is probed -- this is the direct test of whether "discriminator at chance" here means subject
identity has become unrecoverable, or only that the training-time discriminator collapsed.

Dataset: MotionSense, all six activities, 24 subjects, single device (no confound from device or
position). ~20% of subjects (5 of 24) held out entirely.

    python subject_nuisance_experiment.py --lambda_sweep 0 0.05 0.1 0.5 1 2 --seeds 0 1 2
"""
import argparse
import os
import pickle
import warnings

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

from gait_dual_adversarial_v2 import (
    _atomic_dump, aggregate, evaluate_everything, segment_stratified_split, train_model,
)

NPZ = r'har_cache\motionsense_hz50_w2.56_ov0.5_gap1_min4_acts-all.npz'
HELD_OUT_FRAC = 5 / 24          # ~5 of 24 subjects held out entirely
HOLDBACK_FRAC = 0.15            # in-distribution reference: held-back segments within train subjects


def build_split(y_subject_raw, y_activity_raw, seg_id, seed=42):
    subj_enc = LabelEncoder()
    y_subj_all = subj_enc.fit_transform(y_subject_raw)          # every subject, 0..23
    act_enc = LabelEncoder()
    y_act_task = act_enc.fit_transform(y_activity_raw)          # activity is the TARGET, never held out

    rng = np.random.RandomState(seed)
    all_subj = np.arange(len(subj_enc.classes_))
    rng.shuffle(all_subj)
    n_held = max(1, round(HELD_OUT_FRAC * len(all_subj)))
    held_subj = set(all_subj[:n_held].tolist())
    train_subj = sorted(set(all_subj.tolist()) - held_subj)
    print(f"Held-out subjects ({len(held_subj)}): {sorted(subj_enc.classes_[list(held_subj)])}")
    print(f"Training subjects ({len(train_subj)}): {sorted(subj_enc.classes_[train_subj])}")

    # y_subj_nuisance: relabelled 0..len(train_subj)-1 for training subjects, -1 for held-out ones
    remap = {old: new for new, old in enumerate(train_subj)}
    y_subj_nuisance = np.array([remap.get(s, -1) for s in y_subj_all], dtype=np.int64)

    is_train_subj = np.isin(y_subj_all, train_subj)
    pool_idx = np.where(is_train_subj)[0]
    unseen_idx = np.where(~is_train_subj)[0]

    # in-distribution reference: hold back a fraction of SEGMENTS within the training subjects
    segs = np.unique(seg_id[pool_idx])
    rng2 = np.random.RandomState(seed + 1)
    rng2.shuffle(segs)
    n_test = max(1, round(HOLDBACK_FRAC * len(segs)))
    test_segs = set(segs[:n_test].tolist())
    is_test = np.isin(seg_id[pool_idx], list(test_segs))
    train_idx = pool_idx[~is_test]
    heldback_idx = pool_idx[is_test]

    # cell name 'seen_seen' is not a typo: evaluate_everything's --test_cell_probe path only
    # scores the held-out-cell probe on the hardcoded names ('seen_seen', 'unseen_dev_seen_act',
    # 'seen_dev_unseen_act'), so this reuses that path to get the subject probe scored on windows
    # the encoder never saw, exactly as every other condition in this paper is probed.
    cells = {'seen_seen': heldback_idx, 'unseen_subj': unseen_idx}
    n_act_train = len(train_subj)      # nuisance (subject) head size
    n_subj_task = len(act_enc.classes_)  # task (activity) head size
    return (y_subj_nuisance, y_act_task, train_idx, cells, n_act_train, n_subj_task,
            subj_enc, act_enc, held_subj, train_subj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lambda_sweep', type=float, nargs='+', default=[0, 0.05, 0.1, 0.5, 1, 2])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--disc_lr_mult', type=float, default=5.0)
    ap.add_argument('--pooling', type=str, default='stats')
    ap.add_argument('--encoder', type=str, default='cnn')
    ap.add_argument('--disc_hidden', type=int, default=128)
    ap.add_argument('--probe_fit_cap', type=int, default=10000)
    ap.add_argument('--out_dir', type=str, default='./gait_experiment_output_v3_subject_nuisance')
    # discriminator matching (IIDM, Iwasawa et al. 2020) in place of gradient reversal; see run_iidm_native.ps1
    ap.add_argument('--adv_method', choices=['grl', 'iidm'], default='grl')
    ap.add_argument('--iidm_gamma', type=float, default=0.7)
    ap.add_argument('--iidm_constant_lambda', action='store_true')
    ap.add_argument('--iidm_gp', type=float, default=0.0)
    ap.add_argument('--iidm_init_centroids', action='store_true')
    ap.add_argument('--optimizer', choices=['adam', 'rmsprop'], default='adam')
    ap.add_argument('--constant_lr', action='store_true')
    ap.add_argument('--out_name', type=str, default='results_subject_nuisance_motionsense.pkl')
    args = ap.parse_args()
    args.nonlinear_probe = True
    args.knn_probe = True
    args.test_cell_probe = True
    args.disc_heldout = True
    args.mode = 'adversarial'

    os.makedirs(args.out_dir, exist_ok=True)
    dev_torch = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", dev_torch)

    d = np.load(NPZ)
    X = d['X']
    (y_subj_nuisance, y_act_task, train_idx, cells, n_act_train, n_subj_task,
     subj_enc, act_enc, held_subj, train_subj) = build_split(d['y_subject'], d['y_activity'], d['seg_id'])
    seg_id = d['seg_id']
    print(f"X={X.shape}  train windows={len(train_idx):,} ({len(np.unique(seg_id[train_idx]))} segments)  "
          f"activities={list(act_enc.classes_)}  subject chance (nuisance)={1/n_act_train:.4f}")
    for name, idx in cells.items():
        print(f"  cell '{name}': {len(idx):,} windows, {len(np.unique(seg_id[idx])) if len(idx) else 0} segments")

    ckpt = os.path.join(args.out_dir, args.out_name)
    results = pickle.load(open(ckpt, 'rb')) if os.path.exists(ckpt) else {}
    if results:
        print(f"Resumed checkpoint: {list(results.keys())}")

    for lam in args.lambda_sweep:
        key = f"lambda={lam:g}"
        if key in results:
            print(f"Skipping {key} (checkpointed).")
            continue
        print(f"\n{'#'*70}\n{key}\n{'#'*70}")
        runs = []
        for seed in args.seeds:
            # y_subj slot = activity (task, never behind GRL); y_act slot = subject (GRL nuisance)
            model, mean, std, diag = train_model(
                X, y_act_task, y_subj_nuisance, y_subj_nuisance,   # y_dev slot unused (use_dev=False)
                train_idx, n_subj_task, n_act_train, n_act_train,
                use_act=True, use_dev=False, args=args, seed=seed,
                lam_max=lam, mode=args.mode, dev_torch=dev_torch, seg_id=seg_id)
            r = evaluate_everything(model, mean, std, X, y_act_task, y_subj_nuisance, y_subj_nuisance,
                                    seg_id, train_idx, cells, args, dev_torch, seed=seed)
            r.update(diag)
            runs.append(r)
            ho = r.get('probe_activity_tc_seen_seen', float('nan'))
            print(f"  seed {seed}: in-dist(activity)={r['cells']['seen_seen']['bout_acc']:.3f}  "
                  f"unseen_subj(activity)={r['cells']['unseen_subj']['bout_acc']:.3f}  "
                  f"probe_subject(in-pool)={r['probe_activity']:.3f}  probe_subject(held-out)={ho:.3f}  "
                  f"disc_subject={diag['disc_act_acc']:.3f}")
        results[key] = aggregate(runs)
        _atomic_dump(results, ckpt)

    print(f"\nSaved to {ckpt}")


if __name__ == '__main__':
    main()
