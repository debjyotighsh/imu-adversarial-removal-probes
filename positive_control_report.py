import pickle

import numpy as np

r = pickle.load(open('positive_control_results.pkl', 'rb'))


def pm(vals, nd=3):
    v = np.asarray(vals, float)
    m, s = np.nanmean(v), np.nanstd(v, ddof=1)
    return f"{m:.{nd}f}±{s:.{nd}f}"


print("=== LEACE positive control (lambda=0 encoder, 3 seeds) ===")
for tag in ('pre', 'post'):
    lin = [x[f'{tag}_lin'] for x in r['leace']]
    mlp = [x[f'{tag}_mlp'] for x in r['leace']]
    knn = [x[f'{tag}_knn'] for x in r['leace']]
    print(f"  {tag}-LEACE: lin={pm(lin)}  mlp={pm(mlp)}  knn={pm(knn)}")

print("\n=== Selectivity control (lambda=1 encoder, held-out cell, 3 seeds) ===")
for tag in ('real', 'perm'):
    mlp = [x[f'{tag}_mlp'] for x in r['selectivity']]
    knn = [x[f'{tag}_knn'] for x in r['selectivity']]
    lin = [x[f'{tag}_lin'] for x in r['selectivity']]
    print(f"  {tag}: lin={pm(lin)}  mlp={pm(mlp)}  knn={pm(knn)}")
print("  chance = 0.333 (3 ambulatory activity classes)")
