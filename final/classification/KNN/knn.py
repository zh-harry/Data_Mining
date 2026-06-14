import sys
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

train_data = pd.read_csv('../../data/dry_bean_train_std.csv')
test_data = pd.read_csv('../../data/dry_bean_test_std.csv')

Label = 'Class'
features = train_data.columns.drop(Label)

X_train = train_data[features].values
y_train = train_data[Label].values

X_test = test_data[features].values
y_test = test_data[Label].values

k = int(sys.argv[1])

knn_model = KNeighborsClassifier(n_neighbors=k)
knn_model.fit(X_train, y_train)

prob = knn_model.predict_proba(X_test)

df = pd.DataFrame(prob, columns=knn_model.classes_)
df.to_csv('knn_proba_predict.csv', index=False)

max_prob = df.max(axis=1)
best_class = df.idxmax(axis=1)

threshold_prob = float(sys.argv[2])

final_prediction = np.where(max_prob >= threshold_prob, best_class, 'Unknown')

result = test_data.copy()
result['Class'] = final_prediction
result.to_csv('knn_prediction.csv', index=False)

correct_known = np.sum((final_prediction == y_test) & (final_prediction != 'Unknown'))
wrong_known = np.sum((final_prediction != y_test) & (final_prediction != 'Unknown'))
unknown_cnt = np.sum(final_prediction == 'Unknown')
total = len(y_test)

print(f"total samples: {total}")
print(f"correct and known: {correct_known}")
print(f"wrong and known: {wrong_known}")
print(f"unknown: {unknown_cnt}")