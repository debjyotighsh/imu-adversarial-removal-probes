"""
Merge the 2-extra-seed reruns (seeds 3,4; gait_experiment_output_v3_frozen_extra) into the
original 3-seed frozen results (seeds 0,1,2; gait_experiment_output_v3_frozen) for the two
conditions that were extended: HHAR central (dlm5) and RealWorld full pool.

Every "*_raw" list is concatenated (5 seeds). Every base metric that has a "_raw" counterpart is
then recomputed as (mean, population_std) from the merged 5-value list, so it is an exact
5-seed statistic, not a formula combining two summaries. The 'cells' sub-dict (identity metrics)
has no raw values, so those are combined with the exact pooled mean/variance identity for two
known-size groups (n=3, n=2), which is algebraically equal to computing mean/std from all 5 raw
values directly.

Writes merged pkls to gait_experiment_output_v3_frozen_merged5/, leaving both source
directories untouched.

    python merge_extra_seeds.py
"""
import glob
import os
import pickle

import numpy as np

SRC_A = 'gait_experiment_output_v3_frozen'          # seeds 0,1,2
SRC_B = 'gait_experiment_output_v3_frozen_extra'     # seeds 3,4
OUT = 'gait_experiment_output_v3_frozen_merged5'
os.makedirs(OUT, exist_ok=True)


def pooled_mean_std(m1, s1, n1, m2, s2, n2):
    """Exact pooled (mean, population_std) for two groups of known size/mean/pop-std,
    algebraically identical to computing mean/std from the n1+n2 raw values."""
    n = n1 + n2
    m = (n1 * m1 + n2 * m2) / n
    var = (n1 * (s1 ** 2 + (m1 - m) ** 2) + n2 * (s2 ** 2 + (m2 - m) ** 2)) / n
    return m, float(np.sqrt(max(var, 0.0)))


def merge_lambda_dict(va, vb):
    out = {}
    raw_keys = set(k for k in va if k.endswith('_raw')) | set(k for k in vb if k.endswith('_raw'))
    for k in raw_keys:
        ra = va.get(k)
        rb = vb.get(k)
        if k == 'disc_act_conf_raw' or k == 'disc_dev_conf_raw':
            # list of confusion matrices (one per seed) -- just concatenate, no merge needed
            merged = (list(ra) if ra else []) + (list(rb) if rb else [])
        elif ra is not None and rb is not None:
            merged = list(ra) + list(rb)
        else:
            merged = ra if ra is not None else rb
        out[k] = merged

    # recompute every base metric that has a _raw counterpart, from the merged raw list
    for k in list(out.keys()):
        if not k.endswith('_raw'):
            continue
        base = k[:-len('_raw')]
        vals = out[k]
        if base.endswith('_correct') or base.endswith('_n') or base in ('probe_activity_correct', 'probe_activity_n'):
            continue  # counts, no (mean,std) summary needed
        if base.endswith('_conf'):
            continue  # confusion matrices have no scalar summary
        try:
            arr = np.array(vals, dtype=float)
            with np.errstate(invalid='ignore'):
                out[base] = (float(np.nanmean(arr)), float(np.nanstd(arr)))
        except (TypeError, ValueError):
            pass  # not a numeric list (shouldn't happen for the keys above)

    # 'cells' sub-dict: no raw values stored -- pool the two (mean, std) summaries exactly
    if 'cells' in va and 'cells' in vb:
        out['cells'] = {}
        for cell in va['cells']:
            out['cells'][cell] = {}
            for metric in va['cells'][cell]:
                m1, s1 = va['cells'][cell][metric]
                m2, s2 = vb['cells'][cell][metric]
                if np.isnan(m1) and np.isnan(m2):
                    out['cells'][cell][metric] = (float('nan'), float('nan'))
                elif np.isnan(m1) or np.isnan(m2):
                    out['cells'][cell][metric] = (m2, s2) if np.isnan(m1) else (m1, s1)
                else:
                    out['cells'][cell][metric] = pooled_mean_std(m1, s1, 3, m2, s2, 2)
    return out


def merge_file(path_a, path_b, out_path):
    ra = pickle.load(open(path_a, 'rb'))
    rb = pickle.load(open(path_b, 'rb'))
    assert set(ra.keys()) == set(rb.keys()), f"lambda keys differ: {set(ra) ^ set(rb)}"
    merged = {k: merge_lambda_dict(ra[k], rb[k]) for k in ra}
    pickle.dump(merged, open(out_path, 'wb'))
    print(f"wrote {out_path}  ({len(merged)} lambda points, 5 seeds each)")


CONDITIONS = [
    ('hhar_dlm5', 'windows_hz50_win2.56_ov0.5_bike', lambda b: '_cap' not in b and 'dlm5' in b and '_deg' not in b),
    ('realworld_full', 'results_sweep_realworld', lambda b: '_cap' not in b),
]

for name, frag, pred in CONDITIONS:
    hits_a = [p for p in glob.glob(os.path.join(SRC_A, '*.pkl'))
              if frag in os.path.basename(p) and 'segdetail' not in p and 'dynamics' not in p
              and pred(os.path.basename(p))]
    hits_b = [p for p in glob.glob(os.path.join(SRC_B, '*.pkl'))
              if frag in os.path.basename(p) and 'segdetail' not in p and 'dynamics' not in p
              and pred(os.path.basename(p))]
    assert len(hits_a) == 1 and len(hits_b) == 1, (name, hits_a, hits_b)
    out_path = os.path.join(OUT, os.path.basename(hits_a[0]))
    merge_file(hits_a[0], hits_b[0], out_path)
