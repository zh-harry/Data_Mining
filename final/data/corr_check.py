import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 讀取資料集
df = pd.read_csv('dry_bean_train.csv')

# 2. 篩選出純數值欄位 (自動排除字串類型的 Class 欄位)
numeric_df = df.select_dtypes(include=[np.number])

# 3. 計算所有數值特徵的相關係數矩陣
corr_matrix = numeric_df.corr()

plt.figure(figsize=(12, 10))
# 畫出熱力圖，設定顏色區間為 -1 到 1，並顯示數值
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Correlation Matrix of Dry Bean Features')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')