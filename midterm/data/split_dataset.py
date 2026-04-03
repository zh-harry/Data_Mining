import pandas as pd

# read original dataset
original_dataset = pd.read_csv('WineQT.csv')

# 'Id' column is not necessary
original_dataset = original_dataset.drop(columns=['Id'])

# stratified sampling
train_dataset = original_dataset.groupby('quality', group_keys=False).sample(frac=0.8,random_state=1)
test_dataset = original_dataset.drop(train_dataset.index)

# shuffle because data are ordered by 'quality'
train_dataset = train_dataset.sample(frac=1,random_state=1).reset_index(drop=True)
test_dataset = test_dataset.sample(frac=1,random_state=1).reset_index(drop=True)

# turn to csv file
train_dataset.to_csv('train.csv', index=False, header=False)
test_dataset.to_csv('test.csv', index=False, header=False)

# print(train_dataset.shape)
# print(test_dataset.shape)