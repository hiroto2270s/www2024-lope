import importlib, numpy as np
import dataset; importlib.reload(dataset)
from dataset import ACTG175DataLoader
import estimation; importlib.reload(estimation)
run_all = estimation.run_all
loader=ACTG175DataLoader(csv_path="Survival-Analysis-ACTG-175-main/AIDS_ClinicalTrial_GroupStudy175.csv")

n_sims=100
ratio=0.2
hist=int((loader.a_data==0).sum()*ratio)
exp=int((loader.a_data==1).sum()*ratio)
print('hist,exp',hist,exp)
method_names = ["long_term_experiment","long_term_ci","typical_ope_ips","typical_ope_dr","long_term_ope"]
vals = {name: [] for name in method_names}
weight_summaries = []
for i in range(n_sims):
    D_H=loader.generate_dataset(n_data=hist, treatment_group='historical', baseline=True)
    D_E_0=loader.generate_dataset(n_data=exp, treatment_group='experimental', baseline=True)
    D_E_1=loader.generate_dataset(n_data=exp, treatment_group='experimental', baseline=False)
    b_vals, n_vals, comp = run_all(D_H, [D_E_0, D_E_1])
    for name in method_names:
        vals[name].append(n_vals.get(name, float('nan')))
    weight_summaries.append(estimation.get_importance_weight_summary(D_H, D_E_0))

import numpy as np, pprint
stats = {k: (np.nanmean(v), np.nanstd(v)) for k,v in vals.items()}
print('\nEstimator mean/std across sims:')
pprint.pprint(stats)

# Compute variance as in notebook
for k,v in vals.items():
    arr = np.array(v)
    mean = np.nanmean(arr)
    var = np.nanmean((arr-mean)**2)
    sd = np.sqrt(var)
    print(f"{k}: mean={mean:.6f}, var={var:.6e}, sd={sd:.6f}")

# Print first 5 weight summaries
print('\nFirst 5 weight summaries:')
for i,w in enumerate(weight_summaries[:5]):
    print(i)
    for m in method_names:
        mw = w.get(m, None)
        if mw is None:
            print(' ', m, '->', None)
        else:
            print(' ', m, 'baseline mean/std:', mw.get('baseline'), 'new mean/std:', mw.get('new_policy'))

# Check dataset random state
print('\nDataset loader random state repr:')
print(repr(getattr(loader,'random_',None)))
