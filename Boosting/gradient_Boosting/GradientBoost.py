from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import numpy as np

class GradientBoostRegressor:
    def __init__(self, n_estimators=100, max_depth=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.models = []
        self.init_val = None

    def fit(self, X, y):
        self.init_val = np.mean(y)  # f₀(x)
        pred = np.full_like(y, fill_value=self.init_val, dtype=float)

        for i in range(self.n_estimators):
            residual = y - pred
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residual)
            update = tree.predict(X)
            pred += update
            self.models.append(tree)

        self.fitted_pred = pred  # store final prediction for evaluation

    def predict(self, X):
        pred = np.full(X.shape[0], fill_value=self.init_val, dtype=float)
        for tree in self.models:
            pred += tree.predict(X)
        return pred

    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)