import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

train_data = pd.read_csv('../data/train.csv')
test_data = pd.read_csv('../data/test.csv')

Label = 'quality'
Characteristcs = train_data.columns.drop(Label)

X_train = train_data[Characteristcs].values
y_train = train_data[Label].values

X_test = test_data[Characteristcs].values
y_test = test_data[Label].values


for i in range(3,11):
    dt = DecisionTreeClassifier(criterion='gini',
                                max_depth=i,
                                random_state=100)
    dt.fit(X_train,y_train)

    print(f"Train accuracy at {i}: {accuracy_score(y_train,dt.predict(X_train))}")
    print(f"Test accuracy at {i}: {accuracy_score(y_test,dt.predict(X_test))}")