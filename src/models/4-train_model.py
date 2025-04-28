import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle as pkl

with open("models/best_params.pkl", "rb") as f:
    best_params = pkl.load(f)

X_train = pd.read_csv('data/processed_data/X_train_scaled.csv')
y_train = pd.read_csv('data/processed_data/y_train.csv')

clf = RandomForestRegressor(n_jobs=-1, **best_params)

clf.fit(X_train, y_train)

with open("models/trained_model.pkl", 'wb') as f:
    pkl.dump(clf, f)
