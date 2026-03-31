import pandas as pd

df = pd.read_csv('data/training_data_202601.csv')

df_train = df[(df.iloc[:, 0] >= 1) & (df.iloc[:, 0] < 232)]
df_valid = df[(df.iloc[:, 0] >= 232) & (df.iloc[:, 0] < 463)]
df_test  = df[(df.iloc[:, 0] >= 463) & (df.iloc[:, 0] < 563)]

df_train.to_csv('data/training_data_202601_train.csv', index=False)
df_valid.to_csv('data/training_data_202601_val.csv', index=False)
df_test.to_csv('data/training_data_202601_test.csv', index=False)