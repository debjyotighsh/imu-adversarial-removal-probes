"""Factorial test of the moving-target account (output of run_factorial.ps1).

Per cell (mean over three seeds): held-out discriminator (balanced), MLP/linear probe, gap, prediction entropy, realised
representation drift (mean over epochs 2-25 and over epochs 10-25 of the relative drift on 1,024 fixed windows), the ratio of the
adversarial to the identity part of the encoder gradient (the push per update), and identity accuracy. Then marginal effects of each
factor on the gap over the 2x2x2 cells and rank correlations across runs between the gap and (a) drift, (b) push."""
import glob
import pickle

import numpy as np
from scipy.stats import spearmanr

import frozen_analysis as FA

CELLS = ['lr1_k1_r0', 'lr0.1_k1_r0', 'lr1_k5_r0', 'lr0.1_k5_r0', 'lr1_k1_r5', 'lr0.1_k1_r5', 'lr1_k5_r5', 'lr0.1_k5_r5',
         'lr0.1_k1_r0_lam10', 'lr0.1_k5_r0_lam10']


def load(cell):
    d = 'gait_experiment_output_v3_factorial_' + cell
    res = [p for p in glob.glob(d + '/*.pkl') if not p.endswith('_dynamics.pkl')]
    dyn = [p for p in glob.glob(d + '/*_dynamics.pkl')]
    if not res or not dyn:
        return None
    return pickle.load(open(res[0], 'rb')), pickle.load(open(dyn[0], 'rb'))


def per_seed(cell):
    got = load(cell)
    if got is None:
        return None
    r, dyn = got
    key = [k for k in r][0]
    v, dv = r[key], dyn[key]
    n = len(v['disc_act_heldout_tc_seen_seen_raw'])
    rows = []
    for s in range(n):
        recs = dv[s]['dynamics']
        ep = np.array([x['epoch'] for x in recs])
        drift = np.array([x.get('rep_drift', np.nan) for x in recs])
        adv = np.array([x.get('gn_enc_adv_act', np.nan) for x in recs])
        idg = np.array([x.get('gn_enc_id', np.nan) for x in recs])
        conf = v['disc_act_conf_raw'][s]
        rows.append(dict(
            disc=v['disc_act_heldout_tc_seen_seen_raw'][s], lin=v['probe_activity_tc_seen_seen_raw'][s],
            mlp=v['probe_activity_mlp_tc_seen_seen_raw'][s], knn=v['probe_activity_knn_tc_seen_seen_raw'][s],
            ent=FA.conf_stats(conf)[2],
            drift_all=np.nanmean(drift[ep >= 2]), drift_late=np.nanmean(drift[ep >= 10]),
            push=np.nanmean(adv[ep >= 2] / idg[ep >= 2]), push_late=np.nanmean(adv[ep >= 10] / idg[ep >= 10])))
    return rows


def ms(x, nd=3):
    x = np.asarray(x, float)
    return f"{np.nanmean(x):.{nd}f}±{np.nanstd(x, ddof=1) if len(x) > 1 else 0:.{nd}f}"


allrows = {}
print(f"{'cell':<22} {'disc':>13} {'MLP':>7} {'gap(MLP)':>9} {'ent':>5} {'drift 2-25':>11} {'drift 10-25':>12} {'push 2-25':>10} {'push 10-25':>11}")
for c in CELLS:
    rows = per_seed(c)
    if rows is None:
        print(f"{c:<22} (not available)")
        continue
    allrows[c] = rows
    g = [100 * (r['mlp'] - r['disc']) for r in rows]
    print(f"{c:<22} {ms([r['disc'] for r in rows]):>13} {np.mean([r['mlp'] for r in rows]):7.3f} {np.mean(g):9.1f} "
          f"{np.mean([r['ent'] for r in rows]):5.2f} {np.mean([r['drift_all'] for r in rows]):11.3f} "
          f"{np.mean([r['drift_late'] for r in rows]):12.3f} {np.mean([r['push'] for r in rows]):10.3f} {np.mean([r['push_late'] for r in rows]):11.3f}")

fact = [c for c in CELLS[:8] if c in allrows]
if len(fact) == 8:
    print("\nMarginal effect on the gap (points, MLP probe minus held-out discriminator), mean over the four cells at each level:")
    for name, idx in (('encoder lr 0.1 vs 1', 0), ('discriminator steps K=5 vs 1', 1), ('restarts every 5 vs none', 2)):
        hi = [c for c in fact if c.split('_')[idx].replace('lr', '').replace('k', '').replace('r', '') in ('0.1', '5')]
        lo = [c for c in fact if c not in hi]
        gm = lambda cs: np.mean([100 * (r['mlp'] - r['disc']) for c in cs for r in allrows[c]])
        dm = lambda cs: np.mean([r['drift_all'] for c in cs for r in allrows[c]])
        print(f"  {name:<30} gap {gm(hi):6.1f} vs {gm(lo):6.1f}  (change {gm(hi) - gm(lo):+6.1f})   drift {dm(hi):.3f} vs {dm(lo):.3f}")

flat = [(c, r) for c in allrows for r in allrows[c]]
gap = [100 * (r['mlp'] - r['disc']) for _, r in flat]
for lab, key in (('drift (epochs 2-25)', 'drift_all'), ('drift (epochs 10-25)', 'drift_late'),
                 ('push (adv/id gradient, 2-25)', 'push'), ('push (10-25)', 'push_late')):
    print(f"Spearman(gap, {lab}) over {len(flat)} runs: {spearmanr(gap, [r[key] for _, r in flat]).correlation:+.2f}")
