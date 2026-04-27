import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV


train_data = pd.read_csv('../data/train.csv')
test_data = pd.read_csv('../data/test.csv')

label = 'quality'
characteristcs = train_data.columns.drop(label)

X_train = train_data[characteristcs].values
y_train = train_data[label].values

X_test = test_data[characteristcs].values
y_test = test_data[label].values

n_est = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
mx_depth = [int(x) for x in np.linspace(10,110,num=11)]
mx_depth.append(None)
mn_smp_splt = [2,5,10]
mn_smp_lf = [1,2,4]
btstrap = [True,False]

random_grid = {'n_estimators': n_est,
               'max_depth': mx_depth,
               'min_samples_split': mn_smp_splt,
               'min_samples_leaf': mn_smp_lf,
               'bootstrap': btstrap}

RF_model = RandomForestClassifier(criterion='gini',random_state=42)

RF_random = RandomizedSearchCV(estimator=RF_model,
                               param_distributions=random_grid,
                               n_iter=100,
                               cv=3,
                               verbose=2,
                               random_state=42)

result = RF_random.fit(X_train,y_train)

print(f"Best parameters combination:{result.best_params_}")
print(f"=> Best score: {result.best_score_}")

best_model = result.best_estimator_

acc = accuracy_score(y_test, best_model.predict(X_test))
print(f"Best Test Accuracy: {acc}")
