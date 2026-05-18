import pandas as pd

df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')

print("=== TRAIN ===")
print("Shape:", df_train.shape)
print("Regions:", df_train['region_id'].nunique())
print("Columns:", df_train.columns.tolist())
print(df_train[df_train['score'].notna()].head(10).to_csv(index=False))

print("=== TEST ===")
print("Shape:", df_test.shape)
print("Regions:", df_test['region_id'].nunique())