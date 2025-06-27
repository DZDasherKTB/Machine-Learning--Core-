import numpy as np
from collections import Counter

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def _euclidean(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X_test):
        X_test = np.array(X_test)
        predictions = []

        for test_point in X_test:
            distances = [self._euclidean(test_point, x_train) for x_train in self.X_train]
            
            k_indices = np.argsort(distances)[:self.k]
            
            k_labels = self.y_train[k_indices]
            label = Counter(k_labels).most_common(1)[0][0]
            predictions.append(label)

        return np.array(predictions)

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return np.mean(y_pred == y_test)
