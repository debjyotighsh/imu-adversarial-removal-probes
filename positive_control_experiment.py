"""
Two controls for the central HHAR result (Table tab:r1 / Section sec:r1, sec:probes):

1. POSITIVE CONTROL (LEACE). Everything in this paper's central experiments measures
   removal via adversarial training, which only drives a training-time discriminator to
   chance -- a weaker guarantee than a method that provably erases linear structure. If
   our probe pipeline is trustworthy, it should report near-chance LINEAR recoverability
   when a method that actually guarantees that is applied, and it should still find
   NONLINEAR recoverability, replicating for inertial data the same linear/nonlinear gap
   the concept-erasure literature already reports elsewhere (Section sec:erasure_lit).
   We fit LEACE (Belrose et al. 2023, using the authors' own `concept-erasure` package)
   on the frozen, non-adversarially-trained (lambda=0) encoder's training-pool embeddings,
   apply it to the held-out cell, and re-probe.

2. SELECTIVITY CONTROL. A nonlinear probe with enough capacity can read structure that
   isn't really there by fitting incidental patterns in its own fit set (Section 2.5's
   "selectivity" concern, Hewitt & Liang 2019). The standard check is a control task: fit
   the same probe to a version of the label permuted to remove any real relationship with
   the representation, and confirm the probe fails on it. We run this on the central
   lambda=1 condition's held-out embeddings, where the real-label MLP/kNN probes read
   ~0.99 (Table tab:r1) -- if that were a selectivity artefact, the permuted-label probes
   fit on the SAME embeddings would read similarly high. They should not.

Both controls reuse the exact HHAR central condition (ambulatory activities, s3mini and
walk held out, disc_lr_mult=5) and the exact probe machinery (probe_accuracy /
probe_accuracy_mlp / probe_accuracy_knn) used throughout the rest of the paper.

    python positive_control_experiment.py --seeds 0 1 2
"""
import argparse
import pickle

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

from gait_dual_adversarial_v2 import (
    AMBULATORY, _atomic_dump, build_2d_split, predict, probe_accuracy,
    probe_accuracy_knn, probe_accuracy_mlp, segment_stratified_split, train_model,
)

NPZ = r'gait_experiment_output_v2\windows_hz50_win2.56_ov0.5_bike-stairsdown-stairsup-walk_ex-s3mini_2.npz'


class Args:
    epochs = 25
    batch_size = 128
    lr = 1e-3
    disc_lr_mult = 5.0
    pooling = 'stats'
    encoder = 'cnn'
    disc_hidden = 128
    probe_fit_cap = 10000
    mode = 'adversarial'


def run_seed(X, y_subj, y_act, y_dev, train_idx, cells, n_subj, n_act, n_dev,
            lam, seed, dev_torch, args):
    model, mean, std, diag = train_model(
        X, y_subj, y_act, y_dev, train_idx, n_subj, n_act, n_dev,
        use_act=True, use_dev=True, args=args, seed=seed,
        lam_max=lam, mode=args.mode, dev_torch=dev_torch, seg_id=None)
    Xn = (X - mean) / std
    _, emb_train = predict(model, Xn[train_idx], y_subj[train_idx], args.batch_size, dev_torch)
    ho_idx = cells['seen_seen']
    _, emb_ho = predict(model, Xn[ho_idx], y_subj[ho_idx], args.batch_size, dev_torch)
    return emb_train, y_act[train_idx], emb_ho, y_act[ho_idx], diag


def leace_control(emb_train, act_train, emb_ho, act_ho, seed):
    from concept_erasure import LeaceEraser
    Xt = torch.tensor(emb_train, dtype=torch.float32)
    Zt = torch.nn.functional.one_hot(torch.tensor(act_train, dtype=torch.long)).float()
    eraser = LeaceEraser.fit(Xt, Zt)
    emb_train_e = eraser(Xt).numpy()
    emb_ho_e = eraser(torch.tensor(emb_ho, dtype=torch.float32)).numpy()

    fit_mask = segment_stratified_split(np.arange(len(act_train)), act_train, seed=seed)
    out = {}
    for tag, etr, aho in (('pre', emb_train, act_ho), ('post', emb_train_e, act_ho)):
        e = emb_train_e if tag == 'post' else emb_train
        eho = emb_ho_e if tag == 'post' else emb_ho
        out[f'{tag}_lin'] = probe_accuracy(e[fit_mask], act_train[fit_mask], eho, aho)
        out[f'{tag}_mlp'] = probe_accuracy_mlp(e[fit_mask], act_train[fit_mask], eho, aho, seed=seed)
        out[f'{tag}_knn'] = probe_accuracy_knn(e[fit_mask], act_train[fit_mask], eho, aho, seed=seed)
    return out


def selectivity_control(emb_train, act_train, emb_ho, act_ho, seed):
    fit_mask = segment_stratified_split(np.arange(len(act_train)), act_train, seed=seed)
    rng = np.random.RandomState(1000 + seed)
    act_train_perm = rng.permutation(act_train)
    out = {}
    out['real_lin'] = probe_accuracy(emb_train[fit_mask], act_train[fit_mask], emb_ho, act_ho)
    out['real_mlp'] = probe_accuracy_mlp(emb_train[fit_mask], act_train[fit_mask], emb_ho, act_ho, seed=seed)
    out['real_knn'] = probe_accuracy_knn(emb_train[fit_mask], act_train[fit_mask], emb_ho, act_ho, seed=seed)
    out['perm_lin'] = probe_accuracy(emb_train[fit_mask], act_train_perm[fit_mask], emb_ho, act_ho)
    out['perm_mlp'] = probe_accuracy_mlp(emb_train[fit_mask], act_train_perm[fit_mask], emb_ho, act_ho, seed=seed)
    out['perm_knn'] = probe_accuracy_knn(emb_train[fit_mask], act_train_perm[fit_mask], emb_ho, act_ho, seed=seed)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--out', type=str, default='positive_control_results.pkl')
    cli = ap.parse_args()
    args = Args()

    dev_torch = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", dev_torch)

    d = np.load(NPZ, allow_pickle=True)
    X = d['X']
    y_subject, y_activity, y_device = d['y_subject'], d['y_activity'], d['y_device']
    seg_id = d['seg_id']
    keep = np.isin(y_activity, AMBULATORY)
    X, y_subject, y_activity, y_device, seg_id = (
        X[keep], y_subject[keep], y_activity[keep], y_device[keep], seg_id[keep])

    subj_enc = LabelEncoder()
    y_subj = subj_enc.fit_transform(y_subject)
    act_enc = LabelEncoder().fit(sorted(set(y_activity) - {'walk'}))
    dev_enc = LabelEncoder().fit(sorted(set(y_device) - {'s3mini'}))

    def safe_encode(enc, values, held):
        out = np.full(len(values), -1, dtype=np.int64)
        m = values != held
        out[m] = enc.transform(values[m])
        return out

    y_act = safe_encode(act_enc, y_activity, 'walk')
    y_dev = safe_encode(dev_enc, y_device, 's3mini')
    train_idx, cells = build_2d_split(seg_id, y_activity, y_device, 's3mini', 'walk')
    n_subj, n_act, n_dev = len(subj_enc.classes_), len(act_enc.classes_), len(dev_enc.classes_)
    print(f"train windows={len(train_idx):,}  seen_seen={len(cells['seen_seen']):,}")

    results = {'leace': [], 'selectivity': []}
    for seed in cli.seeds:
        print(f"\n--- seed {seed}: training lambda=0 (for LEACE positive control) ---")
        et0, at0, eh0, ah0, _ = run_seed(X, y_subj, y_act, y_dev, train_idx, cells,
                                         n_subj, n_act, n_dev, 0.0, seed, dev_torch, args)
        lc = leace_control(et0, at0, eh0, ah0, seed)
        print(f"  lambda=0, pre-LEACE : lin={lc['pre_lin']:.3f} mlp={lc['pre_mlp']:.3f} knn={lc['pre_knn']:.3f}")
        print(f"  lambda=0, post-LEACE: lin={lc['post_lin']:.3f} mlp={lc['post_mlp']:.3f} knn={lc['post_knn']:.3f}")
        results['leace'].append(lc)

        print(f"\n--- seed {seed}: training lambda=1 (for selectivity control) ---")
        et1, at1, eh1, ah1, _ = run_seed(X, y_subj, y_act, y_dev, train_idx, cells,
                                         n_subj, n_act, n_dev, 1.0, seed, dev_torch, args)
        sc = selectivity_control(et1, at1, eh1, ah1, seed)
        print(f"  lambda=1, real labels : mlp={sc['real_mlp']:.3f} knn={sc['real_knn']:.3f}")
        print(f"  lambda=1, permuted    : mlp={sc['perm_mlp']:.3f} knn={sc['perm_knn']:.3f}  "
              f"(chance={1/n_act:.3f})")
        results['selectivity'].append(sc)

        _atomic_dump(results, cli.out)

    print(f"\nSaved to {cli.out}")


if __name__ == '__main__':
    main()
