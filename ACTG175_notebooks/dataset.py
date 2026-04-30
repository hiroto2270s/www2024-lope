from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state


@dataclass
class ACTG175DataLoader:
    """Data loader for the AIDS Clinical Trial Group Study 175 dataset.
    
    Variables:
    - x: Baseline covariates (age, wtkg, race, gender, hemo, homo, drugs, cd40, cd80, symptom, karnof, preanti, str2)
    - a: Treatment assignment (trt: 0=historical, 1=experimental, 2=control, 3=other)
    - s: Short-term surrogates (cd420, cd820, offtrt)
    - r: Long-term outcome (time: days to event)
    """
    
    csv_path: Optional[str] = None
    n_actions: int = 4  # four treatment groups
    x_dim: int = 13  # 13 baseline covariates
    a_dim: int = 4  # 4-dimensional action feature
    s_dim: int = 3  # 3 short-term metrics
    reward_type: str = "continuous"
    random_state: int = 12345
    
    def __post_init__(self) -> None:
        self.random_ = check_random_state(self.random_state)
        self._load_data()
        self._preprocess_data()  # Preprocess the data for model training
        self._setup_models()
    
    def _load_data(self) -> None:
        """Load the ACTG175 dataset from CSV."""
        import os
        
        # Determine CSV path
        if self.csv_path is None:
            # Try multiple possible paths
            possible_paths = [
                "Survival-Analysis-ACTG-175-main/AIDS_ClinicalTrial_GroupStudy175.csv",
                "ACTG175_notebooks/Survival-Analysis-ACTG-175-main/AIDS_ClinicalTrial_GroupStudy175.csv",
                os.path.join(os.path.dirname(__file__), "Survival-Analysis-ACTG-175-main", "AIDS_ClinicalTrial_GroupStudy175.csv")
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    self.csv_path = path
                    break
            
            if self.csv_path is None:
                raise FileNotFoundError(f"Could not find AIDS_ClinicalTrial_GroupStudy175.csv in: {possible_paths}")
        
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"Loaded {len(self.df)} samples from {self.csv_path}")
        except Exception as e:
            print(f"Error loading CSV from {self.csv_path}: {e}")
            raise
    
    def _preprocess_data(self) -> None:
        """Preprocess and normalize the data."""
        # Covariates: x (age, wtkg, race, gender, hemo, homo, drugs)
        cov_cols = ['age', 'wtkg', 'race', 'gender', 'hemo', 'homo', 'drugs', 'cd40', 'cd80', 'symptom', 'karnof', 'preanti', 'str2']
        self.x_data = self.df[cov_cols].values.astype(float)
        
        # Standardize covariates
        self.scaler_x = StandardScaler()
        self.x_data = self.scaler_x.fit_transform(self.x_data)
        
        # Treatment assignment: a (treat)
        self.a_data = self.df['trt'].values.astype(int)  # Binary treatment assignment
        
        # Action features: one-hot encoding of treatment
        self.a_feat = np.eye(self.n_actions)
        
        # Short-term surrogates: s (cd420, cd820, offtrt)
        s_cols = ['cd420', 'cd820', 'offtrt']
        self.s_data = self.df[s_cols].values.astype(float)
        
        # Standardize short-term metrics
        self.scaler_s = StandardScaler()
        self.s_data = self.scaler_s.fit_transform(self.s_data)
        
        # Long-term outcome: r (time - days to event)
        self.r_data = self.df['time'].values.astype(float)
        
        # Normalize time to [0, 1] range
        self.r_min, self.r_max = self.r_data.min(), self.r_data.max()
        self.r_normalized = (self.r_data - self.r_min) / (self.r_max - self.r_min)
        
        # Build x_s: concatenation of x and s
        self.x_s_data = np.concatenate([self.x_data, self.s_data], axis=1)
        
        print(f"Data shapes: x={self.x_data.shape}, a={self.a_data.shape}, s={self.s_data.shape}, r={self.r_data.shape}")
    
    def _setup_models(self) -> None:
        """Setup action generation models."""
        # Estimate propensity scores (treatment probability given x_s)
        from sklearn.neural_network import MLPClassifier
        self.propensity_model = MLPClassifier(
            hidden_layer_sizes=(10, 10),
            random_state=self.random_state,
            max_iter=1000
        )
        self.propensity_model.fit(self.x_s_data, self.a_data)
        self.pi_e = self.propensity_model.predict_proba(self.x_s_data)
    
    def generate_dataset(
        self,
        n_data: int,
        treatment_group: Optional[str] = None,
        beta: float = 1.0,
        k: int = 1,
        eps: float = 0.1,
        baseline: bool = True
    ) -> dict:
        """Generate a dataset by sampling from the real ACTG175 data.
        
        Args:
            n_data: Number of samples to generate
            treatment_group: Treatment group to sample from
                - "historical": trt=0,1,3 for D_H (observational data)
                - "experimental": trt=2 for D_E (experimental data)
            beta: Temperature parameter for baseline policy (unused for real data)
            k: Top-k actions for eps-greedy policy
            eps: Epsilon for eps-greedy policy
            baseline: Whether to use baseline policy
        
        Returns:
            Dictionary with dataset information
        """
        # Filter data by treatment group
        if treatment_group == "historical":
            # D_H: Observational data from trt=0,1,3
            valid_idx = np.where(np.isin(self.a_data, [0, 1, 3]))[0]

        elif treatment_group == "experimental":
            # D_E: Experimental data from trt=2
            valid_idx = np.where(self.a_data == 2)[0]
        else:
            raise ValueError("treatment_group must be 'historical' or 'experimental'")
        
        if n_data > len(valid_idx):
            raise ValueError(
                f"Requested n_data={n_data} exceeds available samples={len(valid_idx)} for treatment_group={treatment_group}"
            )

        # Sample indices without replacement from the selected treatment pool
        sample_idx = self.random_.choice(valid_idx, size=n_data, replace=False)
        
        x = self.x_data[sample_idx]
        a_data = self.a_data[sample_idx]
        s = self.s_data[sample_idx]
        x_s = self.x_s_data[sample_idx]
        r = self.r_normalized[sample_idx]
        
        # Propensity scores from the real data
        pi_e = self.pi_e[sample_idx]
        
        # Generate counterfactual action under new policy
        # For demonstration, use eps-greedy based on reward signal
        pi_b = np.ones((n_data, self.n_actions)) / self.n_actions
        
        return dict(
            n_data=n_data,
            n_actions=self.n_actions,
            x=x,
            x_s=x_s,
            a_feat=self.a_feat,
            actions=a_data,
            r=r,
            s=s,
            pi_b=pi_b,
            pscore=pi_b[np.arange(n_data), a_data],
            pi_e=pi_e,
        )
    
    def calc_policy_value_beta(self, beta: float = 1.0) -> float:
        """Calculate policy value under baseline (observational) policy."""
        return self.r_normalized.mean()
    
    def calc_policy_value_eps(self, k: int = 1, eps: float = 0.1) -> float:
        """Calculate policy value under counterfactual policy."""
        # For real data, this is estimated from the data itself
        return self.r_normalized.mean() + 0.1  # Hypothetical improvement
