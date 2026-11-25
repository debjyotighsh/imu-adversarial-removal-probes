"""Architecture-generality check (output of run_architectures.ps1): CNN (reference, first three seeds of run_selection.ps1),
Transformer and BiLSTM encoders on the central HHAR sweep and MobiAct condition B.

Per encoder and dataset: lambda=1 held-out discriminator (balanced), entropy, linear / MLP / k-NN held-out probes, gap (strongest
probe minus discriminator, points), and the strongest-probe reduction rho at its best case and with lambda chosen on half A and
read on half B (as in selection_report.py)."""
import glob
import pickle

import numpy as np

import frozen_analysis as FA

D = 'gait_experiment_output_v3_'
SETS = {
    'HHAR activity': dict(ref=D + 'select_hhar', refdisc=D + 'disc_heldout', t=D + 'arch_transformer_hhar', l=D + 'arch_lstm_hhar',
                          stem='probe_activity', disc='disc_act', conf='disc_act_conf_raw', c=1 / 3),
    'MobiAct B activity': dict(ref=D + 'select_mobiact_B', refdisc=D + 'restart2_mobiact_base', t=D + 'arch_transformer_mobiact_B', l=D + 'arch_lstm_mobiact_B',
                               stem='probe_activity', disc='disc_act', conf='disc_act_conf_raw', c=1 / 2),
}


def load(d):
    f = glob.glob(d + '/*.pkl')
    return pickle.load(open(f[0], 'rb')) if f else None


def lam(k):
    return float(k.split('=')[1])


def fam(v, stem, suffix=''):
    out = {}
    for fs, n in (('', 'lin'), ('_mlp', 'mlp'), ('_knn', 'knn')):
        out[n] = np.asarray(v[f'{stem}{fs}_tc_seen_seen{suffix}_raw'], float)[:3]
    return out


def curves(r, stem, suffix=''):
    ks = sorted(r, key=lam)
    L = np.array([lam(k) for k in ks])
    arr = {n: np.array([fam(r[k], stem, suffix)[n] for k in ks], float).T for n in ('lin', 'mlp', 'knn')}
    return L, arr


def best(P):
    return np.nanmax(np.stack(list(P.values())), axis=0)


def rho_est(r, stem, c):
    L, full = curves(r, stem)
    _, A = curves(r, stem, '_halfA')
    _, B = curves(r, stem, '_halfB')
    F, Aa, Bb = best(full), best(A), best(B)
    n = F.shape[0]
    bc = np.array([(F[s, 0] - np.nanmin(F[s])) / (F[s, 0] - c) for s in range(n)])
    ist = np.nanargmin(Aa, axis=1)
    sel = np.array([(Bb[s, 0] - Bb[s, ist[s]]) / (Bb[s, 0] - c) for s in range(n)])
    return 100 * bc, 100 * sel


def ms(x, nd=1):
    x = np.asarray(x, float)
    return f"{np.nanmean(x):.{nd}f}±{np.nanstd(x, ddof=1):.{nd}f}"


for name, cfg in SETS.items():
    print(f"\n=== {name} ===")
    print(f"{'encoder':<12} {'disc(lam=1)':>12} {'ent':>5} {'lin':>6} {'MLP':>6} {'kNN':>6} {'gap':>6}   rho_best best / selected")
    for lab, key in (('CNN', 'ref'), ('Transformer', 't'), ('BiLSTM', 'l')):
        r = load(cfg[key])
        if r is None or 'lambda=1' not in r:
            print(f"{lab:<12} (not available)")
            continue
        v = r['lambda=1']
        rd = load(cfg['refdisc']) if key == 'ref' else r
        vd = rd['lambda=1']
        d = np.asarray(vd[f"{cfg['disc']}_heldout_tc_seen_seen_raw"], float)[:3]
        f = fam(v, cfg['stem'])
        st = np.nanmax(np.stack(list(f.values())), axis=0)
        ent = np.mean([FA.conf_stats(cf)[2] for cf in vd[cfg['conf']][:3]])
        bc, sel = rho_est(r, cfg['stem'], cfg['c'])
        print(f"{lab:<12} {ms(d, 3):>12} {ent:5.2f} {f['lin'].mean():6.3f} {f['mlp'].mean():6.3f} {f['knn'].mean():6.3f} "
              f"{100 * (st.mean() - d.mean()):6.1f}   {ms(bc)} / {ms(sel)}")
    print("  per-lambda discriminator and strongest probe:")
    for lab, key in (('CNN', 'ref'), ('Transformer', 't'), ('BiLSTM', 'l')):
        r = load(cfg[key])
        if r is None:
            continue
        row = []
        rd = load(cfg['refdisc']) if key == 'ref' else r
        for k in sorted(r, key=lam):
            v = r[k]
            if k not in rd:
                continue
            d = np.mean(np.asarray(rd[k][f"{cfg['disc']}_heldout_tc_seen_seen_raw"], float)[:3])
            st = np.nanmean(np.nanmax(np.stack(list(fam(v, cfg['stem']).values())), axis=0))
            row.append(f"{lam(k):g}:{d:.2f}/{st:.2f}")
        print(f"    {lab:<12}", '  '.join(row))
