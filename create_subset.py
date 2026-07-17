import pandas as pd
from sklearn.model_selection import train_test_split

print("Loading original dataset...")
df = pd.read_csv('creditcard.csv')

print("Taking a 10% stratified sample...")
# We use train_test_split to easily get a 10% stratified sample
_, df_10percent = train_test_split(df, test_size=0.10, stratify=df['Class'], random_state=42)

print(f"Original dataset size: {len(df)}")
print(f"New 10% dataset size: {len(df_10percent)}")
print(f"Fraud cases in 10% dataset: {df_10percent['Class'].sum()}")

df_10percent.to_csv('creditcard_10percent.csv', index=False)
print("Saved to creditcard_10percent.csv")
