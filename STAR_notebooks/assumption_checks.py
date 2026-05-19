"""Assumption checks for ACTG175 dataset.

Provides functions to test:
- Ignorability / balance tests (standardized mean differences)
- Full support / overlap (propensity score distributions, extreme IW)
- Surrogacy / Prentice-like test
- Comparability between observational and experimental data

Example usage:
    python ACTG175_notebooks/assumption_checks.py

The script expects the ACTG175 CSV to be available at
`Survival-Analysis-ACTG-175-main/AIDS_ClinicalTrial_GroupStudy175.csv` and
uses `ACTG175_notebooks.dataset.ACTG175DataLoader` to load / sample data.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
# Import only the regression tools to avoid importing time-series DLLs
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from .dataset import ACTG175DataLoader
except Exception:
    # allow running as script from repo root
    from dataset import ACTG175DataLoader


def compute_smd(df, treatment_col, covariates, group_a, group_b=None):
    """Compute standardized mean differences for covariates.

    If group_b is None, compare group_a vs all others.
    Returns a pandas.DataFrame with columns: covariate, smd, mean_a, mean_b
    """
    if group_b is None:
        mask_a = df[treatment_col] == group_a
        mask_b = df[treatment_col] != group_a
    else:
        mask_a = df[treatment_col] == group_a
        mask_b = df[treatment_col] == group_b

    out = []
    for col in covariates:
        a = pd.to_numeric(df.loc[mask_a, col], errors='coerce').dropna()
        b = pd.to_numeric(df.loc[mask_b, col], errors='coerce').dropna()
        if len(a) < 2 or len(b) < 2:
            smd = np.nan
        else:
            mean_a = a.mean()
            mean_b = b.mean()
            var_a = a.var(ddof=1)
            var_b = b.var(ddof=1)
            pooled = np.sqrt((var_a + var_b) / 2.0) if (var_a + var_b) > 0 else 0.0
            smd = (mean_a - mean_b) / pooled if pooled > 0 else np.nan
        out.append({'covariate': col, 'smd': float(np.abs(smd)) if np.isfinite(smd) else np.nan,
                    'mean_a': float(a.mean()) if len(a) else np.nan,
                    'mean_b': float(b.mean()) if len(b) else np.nan})

    return pd.DataFrame(out).set_index('covariate')


def fit_propensity(X, W, scale=True):
    """Fit a multinomial propensity model P(W|X) and return predicted probs and model."""
    Xc = X.copy()
    if isinstance(Xc, pd.DataFrame):
        Xmat = Xc.values
    else:
        Xmat = np.asarray(Xc)
    if scale:
        scaler = StandardScaler()
        Xmat = scaler.fit_transform(Xmat)
    else:
        scaler = None

    # Try to create a multinomial logistic regression; if the installed sklearn
    # doesn't accept `multi_class`, fall back to One-vs-Rest wrapper.
    try:
        clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=2000)
    except TypeError:
        base = LogisticRegression(solver='lbfgs', max_iter=2000)
        clf = OneVsRestClassifier(base)
    clf.fit(Xmat, W)
    probs = clf.predict_proba(Xmat)
    return probs, clf, scaler


def plot_propensity_distributions(probs, W, labels=None, figsize=(10, 6)):
    """Plot propensity score distributions per treatment (for each action)."""
    n_actions = probs.shape[1]
    df = pd.DataFrame(probs, columns=[f'pi_{i}' for i in range(n_actions)])
    df['W'] = W
    plt.figure(figsize=figsize)
    for a in range(n_actions):
        sns.kdeplot(df.loc[df.W == a, f'pi_{a}'], label=f'action={a}')
    plt.title('Propensity distribution per action (self-propensity)')
    plt.legend()
    plt.show()


def importance_weight_summary(pi_e, pi_b, a):
    """Compute importance weight statistics for observed actions `a`.

    pi_e: (n, k) new policy probs
    pi_b: (n, k) behavior propensities
    a: (n,) observed actions
    Returns dict with mean, std, max, pct>10, pct>100
    """
    n = len(a)
    idx = np.arange(n)
    w = (pi_e[idx, a] + 1e-12) / (pi_b[idx, a] + 1e-12)
    return {
        'mean': float(np.mean(w)),
        'std': float(np.std(w)),
        'max': float(np.max(w)),
        'pct_gt_10': float(np.mean(w > 10)),
        'pct_gt_100': float(np.mean(w > 100)),
    }


def placebo_test(df, outcome_var, treatment_col, covariates):
    """Placebo test: regress a pre-treatment variable on treatment + covariates.
    Expect treatment coefficients near zero.
    Returns fitted results object."""
    y = pd.to_numeric(df[outcome_var], errors='coerce')
    X = pd.get_dummies(df[treatment_col].astype('category'), prefix=treatment_col, drop_first=True)
    if len(covariates) > 0:
        X = pd.concat([X, df[covariates]], axis=1)
    X = add_constant(X)
    mask = y.notnull()
    res = OLS(y.loc[mask], X.loc[mask].astype(float)).fit()
    print('Placebo test outcome:', outcome_var)
    print(res.summary())
    return res


def prentice_test(df, long_term_y, treatment_col, s_vars, covariates):
    """Prentice-like test: regress Y ~ W + S + X and inspect treatment coefficient(s)."""
    y = pd.to_numeric(df[long_term_y], errors='coerce')
    W_dummies = pd.get_dummies(df[treatment_col].astype('category'), prefix=treatment_col, drop_first=True)
    X = pd.concat([W_dummies, df[s_vars], df[covariates]], axis=1)
    X = add_constant(X)
    mask = y.notnull()
    res = OLS(y.loc[mask], X.loc[mask].astype(float)).fit()
    print('Prentice-like regression for:', long_term_y)
    print(res.summary())
    return res


def comparability_test(df_obs, df_exp, long_term_y, s_vars, covariates):
    """Test whether data source matters: regress Y ~ S + X + P where P indicates source.

    df_obs, df_exp: DataFrames containing same columns; long_term_y may be missing in one.
    """
    obs = df_obs.copy()
    exp = df_exp.copy()
    obs['P_source'] = 0
    exp['P_source'] = 1
    df_all = pd.concat([obs, exp], ignore_index=True, sort=False)

    if long_term_y not in df_all.columns:
        raise ValueError(f'{long_term_y} not found in combined data for comparability test')

    y = pd.to_numeric(df_all[long_term_y], errors='coerce')
    X = pd.concat([df_all[s_vars], df_all[covariates], df_all[['P_source']]], axis=1)
    X = add_constant(X)
    mask = y.notnull()
    res = OLS(y.loc[mask], X.loc[mask].astype(float)).fit()
    print('Comparability test (P_source coefficient):')
    print(res.summary())
    return res


def example_run():
    print('Loading ACTG175 dataset via ACTG175DataLoader...')
    loader = ACTG175DataLoader(csv_path='Survival-Analysis-ACTG-175-main/AIDS_ClinicalTrial_GroupStudy175.csv')
    # use the full DataFrame and treatment vector
    df = loader.df.copy()
    df['W'] = loader.a_data

    # choose covariates (try to use available numeric baseline covariates)
    possible_covs = ['age', 'wtkg', 'hemo', 'karnof', 'cd40', 'cd80']
    covs = [c for c in possible_covs if c in df.columns]
    print('Using covariates:', covs)

    # 1) Balance: pairwise SMD comparing trt=2 (experimental) vs others
    smd_df = compute_smd(df, 'W', covs, group_a=2, group_b=None)
    print('\nStandardized mean differences (trt=2 vs others):')
    print(smd_df.round(4).to_string())

    # 2) Propensity / overlap
    X = df[covs]
    W = df['W'].values
    pi_b, model, scaler = fit_propensity(X, W)
    print('\nPropensity model fitted. Showing mean probs per action:')
    print(pd.DataFrame(pi_b, columns=[f'pi_b_{i}' for i in range(pi_b.shape[1])]).mean().round(4).to_string())
    try:
        plot_propensity_distributions(pi_b, W)
    except Exception:
        pass

    # 3) IW summary using loader.pi_e (bias policy) if present
    if hasattr(loader, 'pi_e'):
        pi_e = loader.pi_e
        iw_summary = importance_weight_summary(pi_e, pi_b, W)
        print('\nImportance weight summary for loader.pi_e vs estimated pi_b:')
        print(iw_summary)

    # 4) Placebo test: pick a pre-treatment variable (age or wtkg)
    placebo_var = 'age' if 'age' in df.columns else (covs[0] if covs else None)
    if placebo_var is not None:
        _ = placebo_test(df, placebo_var, 'W', covs)

    # 5) Prentice-like test: requires long-term outcome column name; try common names
    possible_long = ['surv_time', 'time', 'y', 'long_term']
    long_found = None
    for name in possible_long:
        if name in df.columns:
            long_found = name
            break
    # fallback: use loader.r (if present) as long-term reward
    if long_found is None and hasattr(loader, 'r_data'):
        df['Y'] = loader.r_data
        long_found = 'Y'
    elif long_found is None and hasattr(loader, 'r'):
        df['Y'] = loader.r
        long_found = 'Y'

    if long_found is not None:
        # short-term surrogates: try common names
        s_vars = [c for c in ['cd420', 'cd820', 'offtrt', 'cd40', 'cd80'] if c in df.columns]
        _ = prentice_test(df, long_found, 'W', s_vars, covs)
    else:
        print('\nLong-term outcome not found in dataset; skip Prentice test.')

    # 6) Comparability: split observational vs experimental by the loader's treatment groups
    # Use generate_dataset to produce two sampled sets with baseline outcome if available
    try:
        D_H = loader.generate_dataset(n_data=200, treatment_group='historical', baseline=True)
        D_E = loader.generate_dataset(n_data=200, treatment_group='experimental', baseline=True)
        # build DataFrames
        df_H = pd.DataFrame(D_H['x']) if 'x' in D_H else pd.DataFrame(loader.x_data)
        df_H['Y'] = D_H.get('r', None)
        df_E = pd.DataFrame(D_E['x']) if 'x' in D_E else pd.DataFrame(loader.x_data)
        df_E['Y'] = D_E.get('r', None)
        # keep short-term s columns if present
        s_cols = []
        if 's' in D_H:
            s_cols = [f's_{i}' for i in range(D_H['s'].shape[1])] if isinstance(D_H['s'], np.ndarray) else []
        # try comparability if Y present in both
        if 'Y' in df_H.columns and 'Y' in df_E.columns:
            _ = comparability_test(df_H, df_E, 'Y', s_cols, list(df_H.columns.difference(['Y'])))
        else:
            print('\nCould not perform comparability test: long-term Y not present in both splits.')
    except Exception as e:
        print('\nSkipping comparability sample-based test due to:', e)


if __name__ == '__main__':
    example_run()
