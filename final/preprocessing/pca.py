import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. 讀取原始資料
input_csv = "../data/dry_bean_train.csv"
print(f"讀取原始資料: {input_csv}")
df = pd.read_csv(input_csv)

# 2. 分離特徵與標籤
# 假設標籤欄位名稱為 'Class'
X = df.drop(columns=['Class'])
y = df['Class']
original_features_count = X.shape[1]

# 3. 特徵標準化 (StandardScaler)
# 這是執行 PCA 前必須進行的步驟，確保各維度尺度一致
print("執行特徵標準化...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. 執行 PCA 並精準測量時間
target_components = 10
pca = PCA(n_components=target_components, random_state=42)

print(f"開始執行 PCA 降維 (目標維度: {target_components})...")

# 記錄開始時間
start_time = time.time()

# 執行模型擬合與資料轉換
X_pca = pca.fit_transform(X_scaled)

# 記錄結束時間
end_time = time.time()

# 計算並儲存執行時間
pca_duration = end_time - start_time

# 5. 組合降維後的資料與原始標籤
df_pca = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(target_components)])
df_final = pd.concat([df_pca, y.reset_index(drop=True)], axis=1)

# 6. 輸出處理後的檔案
output_csv = "../data/dry_bean_train_std_pca.csv"
df_final.to_csv(output_csv, index=False)

# 7. 印出執行報告
print("\n前處理執行完畢。")
print("-" * 40)
print(f"資料維度變化: {original_features_count} 維 -> {target_components} 維")
print(f"PCA 演算法運算時間: {pca_duration:.6f} 秒")
print(f"處理後資料已儲存為: {output_csv}")
print("-" * 40)