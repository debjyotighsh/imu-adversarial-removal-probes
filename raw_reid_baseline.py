"""Raw-signal re-identification baseline for the role-swap experiment (Section "Swapping the Roles").

Same attacker, same labelled pool, same held-back cell and the same three probe families as the paper's
held-out probe in subject_nuisance_experiment.py, but fitted on features of the RAW windows instead of the
frozen encoder's embedding. Answers: how much of the subject-identification accuracy seen in the encoder
(0.59-0.62 at every lambda, chance 0.053) is already available from the signal itself?

Feature sets, all computed on the window standardised with the training pool's per-channel mean/SD:
  stats    mean, max and SD over time per channel (18 values; the same pooling the encoder applies to its
           feature maps, applied to the signal)
  spectral stats + log-magnitude spectrum of each channel averaged into 16 bands (114 values)
  flat     the whole window, 128 x 6 = 768 values

    python raw_reid_baseline.py
"""
import argparse
import pickle

import numpy as np

from gait_dual_adversarial_v2 import probe_accuracy, probe_accuracy_knn, probe_accuracy_mlp
from subject_nuisance_experiment import NPZ, build_split


def feats(Xn, kind):
    st = np.concatenate([Xn.mean(1), Xn.max(1), Xn.std(1)], axis=1)            # (n, 18)
    if kind == 'stats':
        return st
    if kind == 'flat':
        return Xn.reshape(len(Xn), -1)
    if kind == 'spectral':
        F = np.abs(np.fft.rfft(Xn - Xn.mean(1, keepdims=True), axis=1))        # (n, 65, 6)
        F = np.log1p(F[:, 1:65, :])                                             # drop DC, 64 bins
        bands = F.reshape(len(Xn), 16, 4, Xn.shape[2]).mean(2)                  # (n, 16, 6)
        return np.concatenate([st, bands.reshape(len(Xn), -1)], axis=1)
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--probe_fit_cap', type=int, default=10000)
    ap.add_argument('--out', default='raw_reid_baseline.pkl')
    args = ap.parse_args()

    d = np.load(NPZ)
    X = d['X'].astype(np.float32)
    (y_subj_nuis, y_act_task, train_idx, cells, n_nuis, n_task, subj_enc, act_enc, held, train_subj
     ) = build_split(d['y_subject'], d['y_activity'], d['seg_id'])
    idx_c = cells['seen_seen']
    keep = y_subj_nuis[idx_c] >= 0
    idx_c = idx_c[keep]
    print(f"pool={len(train_idx):,} windows, held-back cell={len(idx_c):,} windows, "
          f"{len(np.unique(d['seg_id'][idx_c]))} segments, {n_nuis} subjects, chance={1 / n_nuis:.4f}")

    mean = X[train_idx].reshape(-1, X.shape[-1]).mean(0)
    std = X[train_idx].reshape(-1, X.shape[-1]).std(0) + 1e-8
    Xn = (X - mean) / std

    res = {}
    for kind in ('stats', 'spectral', 'flat'):
        Ftr = feats(Xn[train_idx], kind)
        Fc = feats(Xn[idx_c], kind)
        ytr, yc = y_subj_nuis[train_idx], y_subj_nuis[idx_c]
        out = {'lin': [], 'mlp': [], 'knn': []}
        for seed in args.seeds:
            sel = np.random.RandomState(seed).choice(len(Ftr), min(args.probe_fit_cap, len(Ftr)), replace=False)
            e, l = Ftr[sel], ytr[sel]
            out['lin'].append(probe_accuracy(e, l, Fc, yc))
            out['mlp'].append(probe_accuracy_mlp(e, l, Fc, yc, seed=seed))
            out['knn'].append(probe_accuracy_knn(e, l, Fc, yc, seed=seed))
        res[kind] = {k: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, v)
                     for k, v in out.items()}
        print(f"{kind:9s} dim={Ftr.shape[1]:4d}  " + "  ".join(
            f"{k}={res[kind][k][0]:.3f}±{res[kind][k][1]:.3f}" for k in ('lin', 'mlp', 'knn')))
    res['_meta'] = dict(n_pool=len(train_idx), n_cell=len(idx_c), n_subjects=n_nuis, chance=1 / n_nuis)
    pickle.dump(res, open(args.out, 'wb'))
    print('saved', args.out)


if __name__ == '__main__':
    main()
