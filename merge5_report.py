"""
Every LaTeX value that changes once HHAR-central and RealWorld-full go to 5 seeds. n is
determined per-value from the length of the underlying raw array (5 for these two conditions,
never assumed globally), so nothing here can silently apply an n=3 conversion factor to 5-seed
data or vice versa.

    python merge5_report.py
"""
import pickle

import numpy as np

import seed_stats as SS

MERGED = 'gait_experiment_output_v3_frozen_merged5'
HHAR = pickle.load(open(
    f'{MERGED}/results_sweep_windows_hz50_win2.56_ov0.5_bike-stairsdown-stairsup-walk_ex-s3mini_2_'
    'dev-s3mini_act-walk_hz50_win2.56_acts-all_ex-none_stats_adversarial_dlm5_ep25_seeds0-1-2.pkl', 'rb'))
RW = pickle.load(open(
    f'{MERGED}/results_sweep_realworld_dev-waist_act-walk_hz50_win2.56_acts-all_ex-none_'
    'stats_adversarial_dlm5_ep25_seeds0-1-2.pkl', 'rb'))


def pm(vals, nd=3):
    v = np.asarray(vals, float)
    n = len(v)
    m, s = np.nanmean(v), np.nanstd(v, ddof=1)
    s_txt = f"{s:.{nd}f}" if round(s, nd) > 0 else "{<}" + f"{10 ** -nd:.{nd}f}"
    return f"${m:.{nd}f}\\pm{s_txt}$", n


def pmp(vals, nd=1):
    txt, n = pm(np.asarray(vals, float) * 100, nd)
    return txt, n


print('=' * 70)
print('TABLE 1 (tab:r1) -- HHAR sweep, 5 seeds')
print('=' * 70)
for k in sorted(HHAR, key=lambda k: float(k.split('=')[1])):
    v = HHAR[k]
    disc, n1 = pm(v['disc_act_acc_raw'])
    ho, n2 = pm(v['probe_activity_tc_seen_seen_raw'])
    ip, n3 = pm(v['probe_activity_raw'])
    cdm, cds = v['cells']['unseen_dev_seen_act']['bout_acc']
    idm, ids = v['cells']['seen_seen']['bout_acc']
    n_id = 5
    cd = pm([cdm], 3)  # placeholder; real conversion below since no raw
    F5 = np.sqrt(n_id / (n_id - 1))
    cd_txt = f"${cdm:.3f}\\pm{'{<}0.001' if round(cds*F5,3)==0 else f'{cds*F5:.3f}'}$"
    id_txt = f"${idm:.3f}\\pm{'{<}0.001' if round(ids*F5,3)==0 else f'{ids*F5:.3f}'}$"
    print(f"{float(k.split('=')[1]):.2f} & {disc}(n={n1}) & {ho}(n={n2}) & {ip}(n={n3}) & {cd_txt} & {id_txt} \\\\")

print()
print('=' * 70)
print('TABLE 5 (tab:probefamilies) rows -- HHAR activity/device, RealWorld 102.9k activity/position')
print('=' * 70)
for label, r, stem, c in (('HHAR activity', HHAR, 'probe_activity', 1/3),
                          ('HHAR device', HHAR, 'probe_device', 1/2),
                          ('RealWorld 102.9k activity', RW, 'probe_activity', 1/3),
                          ('RealWorld 102.9k position', RW, 'probe_device', 1/6)):
    a = SS.analyse(r, stem, c, tc='seen_seen')
    n = len(a['p0'])
    cells = [pm(a['p0'])[0]] + [pmp(a['rho_' + k])[0] for k in ('lin', 'mlp', 'knn', 'best')]
    cells.append(pm(a['Rnl'], 2)[0])
    print(f"{label:28s} n={n} & " + ' & '.join(cells) + f" & {', '.join(f'{x:g}' for x in a['lam_star'])} \\\\")

print()
print('=' * 70)
print('TABLE 7 (tab:decodability) row -- HHAR ambulatory')
print('=' * 70)
a = SS.analyse(HHAR, 'probe_activity', 1/3, tc='seen_seen')
print(f"n={len(a['p0'])}  p0 {pm(a['p0'])[0]}  lin {pmp(a['rho_lin'])[0]}  best {pmp(a['rho_best'])[0]}")

print()
print('=' * 70)
print('TABLE 10 (tab:r4) -- RealWorld full sweep, 5 seeds')
print('=' * 70)
for k in sorted(RW, key=lambda k: float(k.split('=')[1])):
    v = RW[k]
    pa, n1 = pm(v['probe_activity_tc_seen_seen_raw'])
    pp, n2 = pm(v['probe_device_tc_seen_seen_raw'])
    da, n3 = pm(v['disc_act_acc_raw'])
    dp, n4 = pm(v['disc_dev_acc_raw'])
    print(f"{float(k.split('=')[1]):.2f} & {pa} & {pp} & {da} & {dp} \\\\   (n={n1},{n2},{n3},{n4})")

print()
print('=' * 70)
print('TABLE 3/9 (tab:r5 / tab:r3) -- RealWorld full-pool identity column, 5 seeds')
print('=' * 70)
n_id = 5
F5 = np.sqrt(n_id / (n_id - 1))
for cell, label in (('seen_seen', 'In-distribution identity'), ('unseen_dev_seen_act', 'Cross-position identity')):
    m, s = RW['lambda=0']['cells'][cell]['bout_acc']
    s_adj = s * F5
    s_txt = f"{s_adj:.3f}" if round(s_adj, 3) > 0 else "{<}0.001"
    print(f"{label:28s}: ${m:.3f}\\pm{s_txt}$  (n={n_id})")

print()
print('=' * 70)
print('Effective band recompute, RealWorld full, 5 seeds (Section r4 text)')
print('=' * 70)
lams = sorted(RW, key=lambda k: float(k.split('=')[1]))
for stem, cname in (('probe_activity', 'activity'), ('probe_device', 'position')):
    lam_f = [float(k.split('=')[1]) for k in lams]
    means = [np.nanmean(RW[k][f'{stem}_tc_seen_seen_raw']) for k in lams]
    p0, pmin = means[0], min(means)
    lmin = lam_f[int(np.argmin(means))]
    mid = (p0 + pmin) / 2
    band = [l for l, x in zip(lam_f, means) if x < mid]
    print(f"{cname}: p0={p0:.3f} min={pmin:.3f} at lambda={lmin:g}  midpoint={mid:.3f}  band={band}")
