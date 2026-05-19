from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from utils import (
    sample_action_fast,
    softmax,
    eps_greedy_policy,
    sigmoid
)


@dataclass
class STARDataLoader:
    """Data loader for the public STAR dataset."""

    csv_path: Optional[str] = None
    n_actions: int = 3
    x_dim: int = 10
    a_dim: int = 3
    s_dim: int = 2
    reward_type: str = "continuous"
    random_state: int = 12345
    preprocess_model: str = "gbdt"  # 'gbdt' or 'linear'
    rare_threshold: float = 0.01
    propensity_model_type: str = "mlp"

    def __post_init__(self) -> None:
        self.random_ = check_random_state(self.random_state)
        self._load_data()
        self._preprocess_data()
        self._fit_surface_models()

    def _load_data(self) -> None:
        """Load STAR.csv from known locations."""
        import os

        if self.csv_path is None:
            possible_paths = [
                "STAR/STAR.csv",
                "ACTG175_notebooks/STAR/STAR.csv",
                os.path.join(os.path.dirname(__file__), "STAR", "STAR.csv"),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    self.csv_path = path
                    break
            if self.csv_path is None:
                raise FileNotFoundError(f"Could not find STAR.csv in: {possible_paths}")

        self.df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(self.df)} samples from {self.csv_path}")

    def _preprocess_data(self) -> None:
        """Prepare x, a, s, r from STAR public data."""
        drop_cols = [
            "gender", "ethnicity", "schoolk", "degreek", "experiencek",
            "ladderk", "tethnicityk", "read3", "math3", "stark",
        ]
        self.df = self.df.dropna(subset=drop_cols).copy()

        # action mapping: regular=0, regular+aide=1, small=2
        self.df["stark"] = self.df["stark"].map({"regular": 0, "regular+aide": 0, "small": 1})

        # long-term reward for STAR setting
        self.df["total3"] = self.df["read3"] + self.df["math3"]

        cov_cols_discrete = ["gender", "ethnicity", "schoolk", "degreek", "ladderk", "tethnicityk"]
        #cov_cols_continuous = ["experiencek"]
        cov_cols_continuous = []

        def _group_rare(series: pd.Series, thresh: float) -> pd.Series:
            if thresh <= 0:
                return series
            vc = series.value_counts(normalize=True)
            rare = vc[vc < thresh].index
            return series.where(~series.isin(rare), other="other")

        for col in cov_cols_discrete:
            self.df[col] = self.df[col].astype(str).str.strip().str.lower()
            self.df[col] = _group_rare(self.df[col], self.rare_threshold)

        if str(self.preprocess_model).lower() == "gbdt":
            self.cat_mappings = {}
            for col in cov_cols_discrete:
                cat = pd.Categorical(self.df[col])
                self.cat_mappings[col] = {v: i for i, v in enumerate(cat.categories)}
                self.df[col] = cat.codes.astype(int)

            x_discrete = self.df[cov_cols_discrete].values.astype(float)
            x_continuous = self.df[cov_cols_continuous].values.astype(float)
            self.x_data = np.concatenate([x_discrete, x_continuous], axis=1)
        else:
            x_discrete = pd.get_dummies(self.df[cov_cols_discrete], drop_first=True)
            self.scaler_x = StandardScaler()
            x_continuous = self.scaler_x.fit_transform(self.df[cov_cols_continuous].values.astype(float))
            x_continuous = pd.DataFrame(x_continuous, index=self.df.index, columns=cov_cols_continuous)
            self.x_data = pd.concat([x_discrete, x_continuous], axis=1).values.astype(float)

        self.x_dim = self.x_data.shape[1]
        self.a_data = self.df["stark"].values.astype(int)
        self.a_feat = np.eye(self.n_actions)

        # short-term surrogates in STAR setting
        #s_cols = ["readk", "mathk"]
        s_cols = ['experiencek']
        self.s_data = self.df[s_cols].values.astype(float)
        self.scaler_s = StandardScaler()
        self.s_data = self.scaler_s.fit_transform(self.s_data)
        self.s_dim = self.s_data.shape[1]

        self.r_data = self.df["total3"].values.astype(float)

        self.x_s_data = np.concatenate([self.x_data, self.s_data], axis=1)
        print(f"Data shapes: x={self.x_data.shape}, a={self.a_data.shape}, s={self.s_data.shape}, r={self.r_data.shape}")

    def _fit_surface_models(self) -> None:
        """Fit proxy surfaces for STAR q(x,a) and f(x,a).

        q_x_a is the estimated conditional mean of the long-term outcome.
        f_x_a is the estimated conditional mean of the short-term surrogate vector.
        """
        strategy = str(self.preprocess_model).lower()
        if strategy == "gbdt":
            q_model = RandomForestRegressor(n_estimators=100,max_depth=5,min_samples_leaf=30,random_state=12345,)
            f_model = RandomForestRegressor(n_estimators=300,max_depth=8,min_samples_leaf=10,random_state=12345,)
        else:
            q_model = MLPRegressor(hidden_layer_sizes=(64, 64), random_state=self.random_state, max_iter=500)
            f_model = MLPRegressor(hidden_layer_sizes=(64, 64), random_state=self.random_state, max_iter=500)

        a_onehot = self.a_feat[self.a_data]
        x_a = np.concatenate([self.x_data, a_onehot], axis=1)

        q_model.fit(x_a, self.r_data)
        f_model.fit(x_a, self.s_data)

        x_rep = np.repeat(self.x_data, self.n_actions, axis=0)
        a_rep = np.tile(self.a_feat, (len(self.df), 1))
        x_a_all = np.concatenate([x_rep, a_rep], axis=1)

        q_pred = q_model.predict(x_a_all)
        if q_pred.ndim == 1:
            q_pred = q_pred[:, np.newaxis]
        self.q_x_a_all = q_pred.reshape(len(self.df), self.n_actions)
        self.q_x_a_factual_all = self.q_x_a_all[np.arange(len(self.df)), self.a_data]

        f_pred = f_model.predict(x_a_all)
        if f_pred.ndim == 1:
            f_pred = f_pred[:, np.newaxis]
        self.f_x_a_all = f_pred.reshape(len(self.df), self.n_actions, self.s_dim)
        self.f_x_a_factual_all = self.f_x_a_all[np.arange(len(self.df)), self.a_data, :]

    def generate_dataset(
        self,
        n_data: int,
        k: int = 1,
        eps: float = 0.1,
        beta: float = 0.0,
        baseline: bool = True,
    ) ->  np.ndarray:
        """Generate STAR historical/experimental split sample."""
        sample_idx = self.random_.choice(len(self.df), size=n_data, replace=True)

        x = self.x_data[sample_idx]
        actions = self.a_data[sample_idx]
        s = self.s_data[sample_idx]
        r = self.r_data[sample_idx]

        q_x_a = self.q_x_a_all[sample_idx]
        q_x_a_factual = q_x_a[np.arange(n_data), actions]
        f_x_a = self.f_x_a_all[sample_idx]
        f_x_a_factual = f_x_a[np.arange(n_data), actions, :]

        if baseline:
            pi_b = softmax(beta * q_x_a)
        else:
            pi_b = eps_greedy_policy(q_x_a, k, eps)

        return dict(
            n_data=n_data,
            n_actions=self.n_actions,
            x=x,
            x_s=np.concatenate([x, s], 1),
            a_feat=self.a_feat,
            actions=actions,
            r=r,
            s=s,
            q_x_a=q_x_a,
            q_x_a_factual=q_x_a_factual,
            f_x_a=f_x_a,
            f_x_a_factual=f_x_a_factual,
            pi_b=pi_b,
            pscore=pi_b[np.arange(n_data), actions],
            pi_e=eps_greedy_policy(q_x_a, k, eps),
        )
    
    def calc_policy_value(self, q_x_a: np.ndarray, pi: np.ndarray) -> float:
        """Calculate the ground-truth expected long-term reward (performance) of a given policy."""
        return (q_x_a * pi).sum(1).mean()


    def calc_policy_value_beta(self, beta: float = 1.0) -> float:
        pi = softmax(beta * self.q_x_a_all)

        return (self.q_x_a_all * pi).sum(1).mean()

    def calc_policy_value_eps(self, k: int = 1, eps: float = 0.1) -> float:
        pi = eps_greedy_policy(self.q_x_a_all, k, eps)

        return (self.q_x_a_all * pi).sum(1).mean()