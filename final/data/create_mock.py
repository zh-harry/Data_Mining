import pandas as pd
from sklearn.model_selection import train_test_split
import os

# 1. 讀取原始訓練集 (包含 5 種已知類別)
csv_path = "dry_bean_train_std_pca.csv"
print(f"讀取原始資料: {csv_path}")
df = pd.read_csv(csv_path)

# 2. 定義已知與假未知類別
# 這 3 種是模型訓練時使用的已知類別
known_classes = ['DERMASON', 'SIRA', 'SEKER']

# 這 2 種是模型未見過，用於驗證集測試攔截效果的類別
mock_unknown_classes = ['HOROZ', 'CALI']

print(f"已知類別設定為: {known_classes}")
print(f"假未知類別設定為: {mock_unknown_classes}")

# 3. 分離資料
df_known = df[df['Class'].isin(known_classes)]
df_mock_unknown = df[df['Class'].isin(mock_unknown_classes)]

# 4. 對已知類別進行 80/20 切割 (分層抽樣)
# 確保切出來的資料中，已知類別的比例保持一致
train_known, val_known = train_test_split(
    df_known, 
    test_size=0.2, 
    random_state=42, 
    stratify=df_known['Class']
)

# 5. 組裝最終檔案
# 訓練集：純淨的 3 種已知類別 (80%)
train_mock = train_known.reset_index(drop=True)

# 驗證集：3 種已知類別的 20% ＋ 2 種假未知類別的 100%
# 將資料打亂順序，避免特定類別過度集中
val_mock = pd.concat([val_known, df_mock_unknown], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

# 6. 輸出與存檔
train_file = "train_mock_std_pca.csv"
val_file = "val_mock_std_pca.csv"

train_mock.to_csv(train_file, index=False)
val_mock.to_csv(val_file, index=False)

print("\n資料集切割與存檔完成。")
print(f"訓練集 ({train_file}): 總計 {len(train_mock)} 筆")
print(f"包含類別: {train_mock['Class'].unique().tolist()}")
print(f"驗證集 ({val_file}): 總計 {len(val_mock)} 筆")
print(f"包含類別: {val_mock['Class'].unique().tolist()} (包含 {len(df_mock_unknown)} 筆假未知類別)")
print("後續可將此兩份檔案用於模型訓練與 Optuna 參數搜尋。")