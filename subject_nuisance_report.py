"""Full report on the subject-nuisance (activity target, subject removed) validation run."""
import pickle

import numpy as np

import frozen_analysis as FA
import seed_stats as SS

r = pickle.load(open('gait_experiment_output_v3_subject_nuisance/results_subject_nuisance_motionsense.pkl', 'rb'))


def pm(vals, nd=3):
    v = np.asarray(vals, float)
    m, s = np.nanmean(v), np.nanstd(v, ddof=1)
    return f"{m:.{nd}f}±{s:.{nd}f}"


print(f"{'lambda':>7} {'disc(subj)':>12} {'probe held-out':>16} {'probe in-pool':>16} "
      f"{'unseen_subj(act)':>18} {'in-dist(act)':>14}")
for k in sorted(r, key=lambda k: float(k.split('=')[1])):
    v = r[k]
    lam = float(k.split('=')[1])
    disc = pm(v['disc_act_acc_raw'])
    ho = pm(v.get('probe_activity_tc_seen_seen_raw', [float('nan')]))
    ip = pm(v['probe_activity_raw'])
    us = pm([v['cells']['unseen_subj']['bout_acc'][0]] * 1) if False else f"{v['cells']['unseen_subj']['bout_acc'][0]:.3f}±{v['cells']['unseen_subj']['bout_acc'][1]*np.sqrt(1.5):.3f}"
    ind = f"{v['cells']['seen_seen']['bout_acc'][0]:.3f}±{v['cells']['seen_seen']['bout_acc'][1]*np.sqrt(1.5):.3f}"
    print(f"{lam:7g} {disc:>12} {ho:>16} {ip:>16} {us:>18} {ind:>14}")

print("\n--- rho (held-out probe) and discriminator vs chance ---")
a = SS.analyse(r, 'probe_activity', 1 / 19, tc='seen_seen')
print(f"rho_lin (held-out) = {100*np.nanmean(a['rho_lin']):.1f}±{100*np.nanstd(a['rho_lin'],ddof=1):.1f}%  "
      f"per seed: {np.round(100*a['rho_lin'],1)}")
print(f"rho_best (held-out) = {100*np.nanmean(a['rho_best']):.1f}±{100*np.nanstd(a['rho_best'],ddof=1):.1f}%  "
      f"per seed: {np.round(100*a['rho_best'],1)}")
print(f"lambda* per seed: {a['lam_star']}")

print("\n--- discriminator confusion / entropy at the point of collapse ---")
for k in ('lambda=0', 'lambda=0.02', 'lambda=1', 'lambda=2'):
    for s, cf in enumerate(r[k]['disc_act_conf_raw']):
        acc, bal, ent, top = FA.conf_stats(cf)
        print(f"  {k} seed {s}: acc {acc:.3f} bal {bal:.3f} entropy {ent:.2f} top-share {top:.3f}  (chance {1/19:.4f})")

print("\n--- gap between discriminator and held-out probe, at the strongest-collapse lambda ---")
for k in sorted(r, key=lambda k: float(k.split('=')[1])):
    d = np.array(r[k]['disc_act_acc_raw'])
    p = np.array(r[k].get('probe_activity_tc_seen_seen_raw', [np.nan] * 3))
    print(f"  {k}: disc {d.mean():.3f}  probe(held-out) {np.nanmean(p):.3f}  gap {100*(np.nanmean(p)-d.mean()):.1f} pts")
