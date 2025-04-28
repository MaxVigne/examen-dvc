import pandas as pd
from sklearn.model_selection import train_test_split

# Read raw data
df = pd.read_csv("data/raw_data/raw.csv")

# Split target variable
X = df.drop(columns=["silica_concentrate", "date"])
y = df["silica_concentrate"]

# Split train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Write to csv files
X_train.to_csv("data/processed_data/X_train.csv")
X_test.to_csv("data/processed_data/X_test.csv")
y_train.to_csv("data/processed_data/y_train.csv")
y_test.to_csv("data/processed_data/y_test.csv")