import pandas as pd
import numpy as np
import sys
from sklearn.svm import SVC

train_source = sys.argv[1]
test_source = sys.argv[2]

train_data = pd.read_csv(train_source)
test_data = pd.read_csv(test_source)

Label = 'Class'
Characteristcs = train_data.columns.drop(Label)

X_train = train_data[Characteristcs].values
y_train = train_data[Label].values

X_test = test_data[Characteristcs].values
y_test = test_data[Label].values

kern = sys.argv[3]

svm_model = SVC(decision_function_shape='ovr', kernel=kern, probability=True)

svm_model.fit(X_train, y_train)

prob = svm_model.predict_proba(X_test)

df = pd.DataFrame(prob, columns=svm_model.classes_)
df.to_csv('svm_proba_predict.csv', index=False)

max_prob = df.max(axis=1)
best_class = df.idxmax(axis=1)

threshold_prob = float(sys.argv[4])

final_prediction = np.where(max_prob >= threshold_prob, best_class, 'Unknown')

result = test_data.copy()
result['Class'] = final_prediction
result.to_csv('svm_prediction.csv', index=False)

correct_known = np.sum((final_prediction == y_test) & (final_prediction != 'Unknown'))
wrong_known = np.sum((final_prediction != y_test) & (final_prediction != 'Unknown'))
unknown_cnt = np.sum(final_prediction == 'Unknown')
total = len(y_test)

print(f"total samples: {total}")
print(f"correct and known: {correct_known}")
print(f"wrong and known: {wrong_known}")
print(f"unknown: {unknown_cnt}")