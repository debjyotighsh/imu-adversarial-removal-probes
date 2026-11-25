"""
Formal test of the linear-vs-nonlinear probe gap, replacing the pooled Wilcoxon's implicit
"21-24 independent observations" framing with a mixed-effects model that has the right
dependence structure: probe type and lambda as fixed effects (lambda continuous, as in the
reviewer's own proposed formula), a per-seed random intercept, and a probe-type x lambda
interaction. Tests whether probe type matters at all (2 df: main effect + interaction)
against a lambda-only null, via likelihood-ratio test.

    y_ijk = b0 + b1*ProbeType + b2*lambda + b3*(ProbeType x lambda) + u_seed + eps

    python probetype_interaction_model.py
"""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.mixed_linear_model import MixedLM

from probe_families_table import CONDITIONS, find_pkl
import pickle

D = 'gait_experiment_output_v3_frozen'


def load(frag, pred):
    hits = find_pkl(D, frag, pred)
    return pickle.load(open(hits[0], 'rb')) if hits else None


def long_frame(r, stem):
    lams = sorted(r, key=lambda k: float(k.split('=')[1]))
    rows = []
    for k in lams:
        lam = float(k.split('=')[1])
        l_ = r[k].get(f'{stem}_tc_seen_seen_raw')
        m_ = r[k].get(f'{stem}_mlp_tc_seen_seen_raw')
        if l_ is None or m_ is None:
            continue
        for s, v in enumerate(l_):
            rows.append(dict(acc=v, probe=0, lam=lam, seed=s))
        for s, v in enumerate(m_):
            rows.append(dict(acc=v, probe=1, lam=lam, seed=s))
    return pd.DataFrame(rows)


def interaction_test(r, stem, label):
    df = long_frame(r, stem)
    if df.empty:
        print(f"  [{label}] no data")
        return
    n_seed = df['seed'].nunique()
    n_lam = df['lam'].nunique()
    full = MixedLM.from_formula('acc ~ probe * lam', groups='seed', data=df).fit(reml=False)
    null = MixedLM.from_formula('acc ~ lam', groups='seed', data=df).fit(reml=False)
    lr = 2 * (full.llf - null.llf)
    p = stats.chi2.sf(lr, df=2)   # 2 extra params: probe main effect + probe:lam interaction
    b_probe = full.params['probe']
    print(f"  [{label}] n={len(df)} rows ({n_lam} lambda x {n_seed} seeds x 2 probe types)")
    print(f"    marginal effect of probe type (MLP - linear) at lambda=0: {b_probe:+.4f}")
    print(f"    LR test (probe main effect + interaction) vs lambda-only null: "
          f"LR={lr:.2f}, df=2, p={p:.3g}")


print("=" * 78)
print("Mixed-effects test: ProbeType x lambda, random intercept by seed")
print("=" * 78)
hh = load('windows_hz50_win2.56_ov0.5_bike', lambda b: '_cap' not in b and 'dlm5' in b and '_deg' not in b)
rw = load('results_sweep_realworld', lambda b: '_cap' not in b)
mb = load('results_sweep_mobiact', lambda b: '_cap' not in b)

interaction_test(hh, 'probe_activity', 'HHAR activity')
interaction_test(rw, 'probe_activity', 'RealWorld activity (full pool)')
interaction_test(rw, 'probe_device', 'RealWorld position (full pool)')
interaction_test(mb, 'probe_activity', 'MobiAct stairs')
