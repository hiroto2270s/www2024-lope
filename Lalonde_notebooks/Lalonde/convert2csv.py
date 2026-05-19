import pyreadr
import pandas as pd

# 1. .rdaファイルを読み込む
result = pyreadr.read_r('nsw_dw.rda')

# 2. データフレームを取り出す
# 辞書のキーは通常ファイル名（nsw_dw）になっています
df = result['nsw_dw']

# 3. 一つのCSVファイルとして保存
# index=False を指定して不要な行番号が入らないようにします
df.to_csv('nsw_dw_all.csv', index=False)

print(f"保存完了: 全 {len(df)} 件（介入群: {sum(df['treat']==1)}件, 対照群: {sum(df['treat']==0)}件）")