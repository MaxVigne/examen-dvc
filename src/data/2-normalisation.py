import pandas as pd
import sklearn.preprocessing as preprocessing

X_train = pd.read_csv("data/processed_data/X_train.csv")
X_test = pd.read_csv("data/processed_data/X_test.csv")

scl = preprocessing.StandardScaler()
X_train_scaled = scl.fit_transform(X_train)
X_test_scaled = scl.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled)
X_test_scaled = pd.DataFrame(X_test_scaled)

X_train_scaled.to_csv("data/processed_data/X_train_scaled.csv")
X_test_scaled.to_csv("data/processed_data/X_test_scaled.csv")
