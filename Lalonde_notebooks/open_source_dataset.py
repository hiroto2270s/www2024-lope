import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any

@dataclass
class LalondeDataLoader:
    path: str = "nsw_dw_all.csv"
    
    def __post_init__(self):
        # データのロード
        self.df = pd.read_csv(self.path)
        
        # 特徴量定義
        # x: 共変量, s: 短期指標(20週目CD4/CD8), a: 治療(treat), r: 長期報酬(生存期間)
        self.x_cols = ['age', 'education', 'black', 'hispanic', 'married', 'nodegree']
        self.s_cols = ['re74', 're75']
        self.a_col = 'treat'
        self.y_col = 're78'

        # 欠損値の削除とスケーリング
        self.df = self.df.dropna(subset=self.s_cols + [self.y_col])
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.df[self.x_cols])
        self.S_scaled = self.scaler.fit_transform(self.df[self.s_cols])
        
        self.n_actions = 4
        self.a_feat = np.eye(self.n_actions) # One-hotアクション特徴量
