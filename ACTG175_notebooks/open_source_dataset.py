import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any

@dataclass
class ACTG175DataLoader:
    path: str = "AIDS_ClinicalTrial_GroupStudy175.csv"
    
    def __post_init__(self):
        # データのロード
        self.df = pd.read_csv(self.path)
        
        # 特徴量定義
        # x: 共変量, s: 短期指標(20週目CD4/CD8), a: 治療(trt), r: 長期報酬(生存期間)
        self.x_cols = ['age', 'wtkg', 'hemo', 'homo', 'drugs', 'karnof', 
                       'oprior', 'z30', 'preanti', 'race', 'gender', 'str2', 'symptom']
        self.s_cols = ['cd420', 'cd820']
        self.a_col = 'trt'
        self.y_col = 'time'

        # 欠損値の削除とスケーリング
        self.df = self.df.dropna(subset=self.s_cols + [self.y_col])
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.df[self.x_cols])
        self.S_scaled = self.scaler.fit_transform(self.df[self.s_cols])
        
        self.n_actions = 4
        self.a_feat = np.eye(self.n_actions) # One-hotアクション特徴量
