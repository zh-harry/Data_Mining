import pandas as pd

train = pd.read_csv('../data/dry_bean_train.csv')
test = pd.read_csv('../data/dry_bean_test.csv')

X_train = train.drop(columns=['Class'])
y_train = train['Class']

X_test = test.drop(columns=['Class'])
y_test = test['Class']

mu = X_train.mean(axis=0)
sigma = X_train.std(axis=0)
    
X_train_std = (X_train - mu) / sigma
X_test_std = (X_test - mu) / sigma

train_final = pd.concat([X_train_std, y_train], axis=1)
test_final = pd.concat([X_test_std, y_test], axis=1)

train_final.to_csv('../data/dry_bean_train_std.csv', index=False)
test_final.to_csv('../data/dry_bean_test_std.csv', index=False)
