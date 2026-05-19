from typing import List, Dict, Tuple
import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import KFold
from sklearn.base import clone

def run_long_term_experiment(D_E_list: List[dict]):
    """Evaluate the true expected reward under each policy using the ground-truth r (Oracle).
    """
    estimated_values = dict()
    for i, D_E in enumerate(D_E_list):
        estimated_values[i] = D_E["r"].mean()
    return estimated_values

def run_long_term_ci(    
    D_H: dict, 
    D_E_list: List[dict], 
    g_x_s_model = RandomForestRegressor(n_estimators=100, random_state=12345),
) -> float:
    """Estimate via long-term causal inference (LCI/Surrogate Index)."""
    """Estimate the long-term expected reward under a given policy via long-term causal inference based on the surrogacy assumption."""
    estimated_values = dict()
    g_x_s_model.fit(D_H["x_s"], D_H["r"])
    for i, D_E in enumerate(D_E_list):
        estimated_values[i] = g_x_s_model.predict(D_E["x_s"]).mean()
    
    return estimated_values

def estimate_q_x_a_via_regression(
    D_H: dict,
    q_x_a_model=RandomForestRegressor(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=20,
        max_features="sqrt",
        random_state=12345,
        n_jobs=-1,
    ),
    n_splits: int = 2,
) -> np.ndarray:
    """Estimate q(x,a) with cross-fitting."""

    n_data, n_actions = D_H["n_data"], D_H["n_actions"]

    x = D_H["x"]
    r = D_H["r"]
    actions = D_H["actions"]
    a_feat = D_H["a_feat"]

    # 出力
    q_x_a_hat = np.zeros((n_data, n_actions))

    # KFold
    kf = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=12345,
    )

    for train_idx, test_idx in kf.split(x):

        # train data
        x_train = x[train_idx]
        r_train = r[train_idx]
        a_train = actions[train_idx]

        # action one-hot
        a_onehot_train = a_feat[a_train]

        # [x,a]
        x_a_train = np.concatenate(
            [x_train, a_onehot_train],
            axis=1,
        )

        # clone model
        model = clone(q_x_a_model)

        # fit only train fold
        model.fit(x_a_train, r_train)

        # predict all actions on test fold
        x_test = x[test_idx]

        for a in range(n_actions):

            a_test = np.tile(
                a_feat[a],
                (len(test_idx), 1),
            )

            x_a_test = np.concatenate(
                [x_test, a_test],
                axis=1,
            )

            q_x_a_hat[test_idx, a] = model.predict(
                x_a_test
            )

    return q_x_a_hat


def run_typical_ope(D_H: dict,
    q_x_a_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=5,
    max_features=None,
    bootstrap=True,
    random_state=12345,
    n_jobs=-1,
),
) -> float:
    """Calculate traditional point estimates for baseline and new policy (IPS and DR)."""
    n_data = D_H["n_data"]
    r, actions = D_H["r"], D_H["actions"]
    pi_b = D_H["pi_b"]
    
    q_x_a_hat = estimate_q_x_a_via_regression(D_H, q_x_a_model=q_x_a_model)
    factual_q_x_a_hat = q_x_a_hat[np.arange(n_data), actions]
    
    estimated_values = dict()
    for i, policy in enumerate(["pi_b", "pi_e"]):
        pi_i = D_H[policy]
        iw = pi_i[np.arange(n_data), actions] / pi_b[np.arange(n_data), actions]
        
        # 1. IPS 点推定
        estimated_values[f"ips{i}"] = (iw * r).sum() / iw.sum()
        
        # 2. DR 点推定
        q_x_pi_hat = (q_x_a_hat * pi_i).sum(1)
        iw = np.clip(iw, np.percentile(iw, 1), np.percentile(iw, 99))  # ウェイトのクリッピング
        iw = np.clip(iw, 1e-7, 1e7)  # ウェイトのクリッピング
        iw_norm = iw / iw.mean()  # ウェイトの正規化
        estimated_values[f"dr{i}"] = (iw_norm * (r - factual_q_x_a_hat) + q_x_pi_hat).mean()
    
    return estimated_values

def run_long_term_ope(D_H: dict, D_E_0: dict,
    q_x_a_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=5,
    max_features=None,
    bootstrap=True,
    random_state=12345,
    n_jobs=-1,
),
    pi_a_x_s_model = CalibratedClassifierCV(
    estimator= LogisticRegression(C=0.1,max_iter=1000,random_state=12345,),
    method="sigmoid",
    cv=3,
)) -> float:
    """Estimate the long-term expected reward under a given policy via long-term OPE (ours), which combines short- and long-term rewards in the historical data."""
    n_data = D_H["n_data"]
    x_s, r, actions = D_H["x_s"], D_H["r"], D_H["actions"]
    pi_b = D_H["pi_b"]
    q_x_a_hat = estimate_q_x_a_via_regression(D_H, q_x_a_model=q_x_a_model)
    
    x_s_ = np.concatenate([D_H["x_s"], D_E_0["x_s"]])
    actions_ = np.concatenate([actions, D_E_0["actions"]])
    observed_action_set = np.unique(actions_)
    pi_a_x_s_model.fit(x_s_, actions_)
    pi_a_x_s_hat = np.zeros(shape=(n_data, D_H["n_actions"]))
    pi_a_x_s_hat[:, observed_action_set] = pi_a_x_s_model.predict_proba(x_s)
    
    estimated_values = dict()
    factual_q_x_a_hat = q_x_a_hat[np.arange(n_data), actions]
    for i, policy in enumerate(["pi_b", "pi_e"]):
        pi_i = D_H[policy]
        iw_hat = ((pi_i / pi_b) * pi_a_x_s_hat).sum(1)
        #iw_hat = np.clip(iw_hat, np.percentile(iw_hat, 1), np.percentile(iw_hat, 99))  # ウェイトのクリッピング
        iw_hat = np.clip(iw_hat, 1e-7, 1e7)  # ウェイトのクリッピング
        iw_hat_norm = iw_hat / iw_hat.mean()  # ウェイトの正規化
        q_x_pi_hat = (q_x_a_hat * pi_i).sum(1)
        estimated_values[i] = (iw_hat_norm * (r - factual_q_x_a_hat) + q_x_pi_hat).mean()
    
    return estimated_values


def run_long_term_ope_gmsm(
    D_H: dict, D_E_0: dict, gamma: float = 1.2,
    q_x_a_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=5,
    max_features=None,
    bootstrap=True,
    random_state=12345,
    n_jobs=-1,
),
    pi_a_x_s_model = CalibratedClassifierCV(
    estimator= LogisticRegression(C=0.1,max_iter=1000,random_state=12345,),
    method="sigmoid",
    cv=3,
)) -> float:
    
    """Estimate the sharp lower and upper bounds of LOPE under the Marginal Sensitivity Model (GMSM)."""
    n_data = D_H["n_data"]
    x_s, r, actions = D_H["x_s"], D_H["r"], D_H["actions"]
    pi_b = D_H["pi_b"]
    q_x_a_hat = estimate_q_x_a_via_regression(D_H, q_x_a_model=q_x_a_model)
    
    # 条件付き選択確率 pi(a|x,s) の学習と予測
    x_s_ = np.concatenate([D_H["x_s"], D_E_0["x_s"]])
    actions_ = np.concatenate([actions, D_E_0["actions"]])
    observed_action_set = np.unique(actions_)
    pi_a_x_s_model.fit(x_s_, actions_)
    pi_a_x_s_hat = np.zeros(shape=(n_data, D_H["n_actions"]))
    pi_a_x_s_hat[:, observed_action_set] = pi_a_x_s_model.predict_proba(x_s)
    
    w_min = 1.0 / gamma
    w_max = gamma
    
    bounds = {"lower": {}, "upper": {}}
    factual_q_x_a_hat = q_x_a_hat[np.arange(n_data), actions]
    
    for i, policy in enumerate(["pi_b", "pi_e"]):
        pi_i = D_H[policy]
        iw_hat = ((pi_i / pi_b) * pi_a_x_s_hat).sum(1)
        #iw_hat = np.clip(iw_hat, np.percentile(iw_hat, 1), np.percentile(iw_hat, 99))  # ウェイトのクリッピング
        iw_hat = np.clip(iw_hat, 1e-7, 1e7)  # ウェイトのクリッピング
        iw_hat_norm = iw_hat / iw_hat.mean()  # ウェイトの正規化
        q_x_pi_hat = (q_x_a_hat * pi_i).sum(1)
        
        # 🌟未観測交絡（MSMウェイト）の影響を受ける短期サロゲート残差項 (Term 1)
        base_values = iw_hat_norm * (r - factual_q_x_a_hat)
        # 未観測交絡の影響を受けない決定論的な固定期待値項 (Term 2)
        term2 = q_x_pi_hat.mean()
        
        # 効率的に最悪ケースのウェイトを配分するためのソートパズル
        sorted_indices_desc = np.argsort(base_values)[::-1]
        
        # --- Upper Bound (上限) の計算 ---
        omega_upper = np.full(n_data, w_min)
        remaining_budget_upper = n_data - (n_data * w_min)
        for idx in sorted_indices_desc:
            alloc = min(remaining_budget_upper, w_max - w_min)
            omega_upper[idx] += alloc
            remaining_budget_upper -= alloc
            if remaining_budget_upper <= 0:
                break
        bounds["upper"][i] = (omega_upper * base_values).mean() + term2
        
        # --- Lower Bound (下限) の計算 ---
        omega_lower = np.full(n_data, w_max)
        remaining_budget_lower = (n_data * w_max) - n_data
        for idx in sorted_indices_desc:
            alloc = min(remaining_budget_lower, w_max - w_min)
            omega_lower[idx] -= alloc
            remaining_budget_lower -= alloc
            if remaining_budget_lower <= 0:
                break
        bounds["lower"][i] = (omega_lower * base_values).mean() + term2
        
    return bounds


def run_all(D_H: dict, D_E_list: List[dict], gamma: float = 1.2) -> List[dict]:
    """Run all methods to estimate the long-term expected reward under a given policy simultaneously."""
    long_term_experiment = run_long_term_experiment(D_E_list)
    long_term_ci = run_long_term_ci(D_H, D_E_list)
    typical_ope = run_typical_ope(D_H)
    long_term_ope = run_long_term_ope(D_H, D_E_list[0])
    lope_gmsm_bounds = run_long_term_ope_gmsm(D_H, D_E_list[0], gamma=gamma)
    
    estimated_values_of_baseline = {
        "long_term_experiment": long_term_experiment[0],
        "long_term_ci": long_term_ci[0],
        #"typical_ope_ips": typical_ope["ips0"],
        "typical_ope_dr": typical_ope["dr0"],
        "long_term_ope": long_term_ope[0],
        "proposed_lope_gmsm_lower": lope_gmsm_bounds["lower"][0], # 追加
        "proposed_lope_gmsm_upper": lope_gmsm_bounds["upper"][0]  # 追加
    }
    estimated_values_of_new_policy = {
        "long_term_experiment": long_term_experiment[1],
        "long_term_ci": long_term_ci[1],
        #"typical_ope_ips": typical_ope["ips1"],
        "typical_ope_dr": typical_ope["dr1"],
        "long_term_ope": long_term_ope[1],
        "proposed_lope_gmsm_lower": lope_gmsm_bounds["lower"][1], # 追加
        "proposed_lope_gmsm_upper": lope_gmsm_bounds["upper"][1]  # 追加
    }
    estimated_policy_comparison = dict()
    for method in estimated_values_of_new_policy:
        is_new_policy_better = int(estimated_values_of_baseline[method] < estimated_values_of_new_policy[method])
        estimated_policy_comparison[method] = is_new_policy_better - (1 - is_new_policy_better)
    
    return estimated_values_of_baseline, estimated_values_of_new_policy, estimated_policy_comparison