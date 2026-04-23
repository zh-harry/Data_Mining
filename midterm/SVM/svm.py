import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Continuous

train_data = pd.read_csv('../data/train.csv')
test_data = pd.read_csv('../data/test.csv')

Label = 'quality'
Characteristcs = train_data.columns.drop(Label)

X_train = train_data[Characteristcs].values
y_train = train_data[Label].values

X_test = test_data[Characteristcs].values
y_test = test_data[Label].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

parameters = {'C': Continuous(0.1,10),
              'gamma': Continuous(0.1,10)}

svm_model = SVC(decision_function_shape='ovr', kernel='rbf')

ga = GASearchCV(estimator=svm_model, 
                cv=5,
                scoring='accuracy',
                param_grid=parameters,
                population_size=50,
                generations=10,
                crossover_probability=0.9,
                mutation_probability=0.1,
                tournament_size=3,
                elitism=True,
                verbose=True)

ga_result = ga.fit(X_train,y_train)

print(f"Best parameters combination:{ga_result.best_params_}")
print(f"=> Best score: {ga_result.best_score_}")

best_svm = ga_result.best_estimator_

acc = accuracy_score(y_test, best_svm.predict(X_test))
print(f"Best Test Accuracy: {acc}")
