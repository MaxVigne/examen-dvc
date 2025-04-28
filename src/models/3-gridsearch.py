import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import pickle as pkl

X_train = pd.read_csv('data/processed_data/X_train_scaled.csv')
X_test = pd.read_csv('data/processed_data/X_test_scaled.csv')
y_train = pd.read_csv('data/processed_data/y_train.csv')
y_test = pd.read_csv('data/processed_data/y_test.csv')

clf = RandomForestRegressor(n_jobs=-1)

params = {
    "n_estimators": [100, 200, 400],
    "max_features": ["sqrt", "log2"],
    "criterion": ["squared_error", "friedman_mse"]
}

grid = GridSearchCV(estimator=clf, param_grid=params, scoring='r2', n_jobs=-1)

print("Performing grid search, please wait.")
grid.fit(X_train, y_train)

print("best params: ", grid.best_params_)
print("best score: ", grid.best_score_)
print("score on test:", grid.score(X_test, y_test))

with open("models/best_params.pkl", "wb") as f:
    pkl.dump(grid.best_params_, f)