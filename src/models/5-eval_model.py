import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import pickle as pkl
import json

X_test = pd.read_csv("data/processed_data/X_test_scaled.csv")
y_test = pd.read_csv("data/processed_data/y_test.csv")

with open("models/trained_model.pkl", "rb") as f:
    clf = pkl.load(f)

y_pred = clf.predict(X_test)

metrics = {
    "r2": r2_score(y_test, y_pred),
    "mse": mean_squared_error(y_test, y_pred)
}

pd.DataFrame(y_pred).to_csv("data/processed_data/y_pred.csv")

with open('metrics/scores.json', "w") as f:
    json.dump(metrics, f)