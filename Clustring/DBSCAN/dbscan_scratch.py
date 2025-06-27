import numpy as np
from collections import deque

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def fit(self, X):
        n = X.shape[0]
        self.labels_ = np.full(n, -1)
        cluster_id = 0

        for i in range(n):
            if self.labels_[i] != -1:
                continue

            neighbors = self._region_query(X, i)

            if len(neighbors) < self.min_samples:
                self.labels_[i] = -1 
            else:
                self._expand_cluster(X, i, neighbors, cluster_id)
                cluster_id += 1

    def _region_query(self, X, idx):
        distances = np.linalg.norm(X - X[idx], axis=1)
        return np.where(distances <= self.eps)[0]

    def _expand_cluster(self, X, idx, neighbors, cluster_id):
        queue = deque(neighbors)
        self.labels_[idx] = cluster_id

        while queue:
            point_idx = queue.popleft()
            if self.labels_[point_idx] == -1:
                self.labels_[point_idx] = cluster_id  # noise → cluster
            elif self.labels_[point_idx] != -1:
                continue  # already assigned

            self.labels_[point_idx] = cluster_id
            new_neighbors = self._region_query(X, point_idx)

            if len(new_neighbors) >= self.min_samples:
                queue.extend(new_neighbors)

from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

X, _ = make_moons(n_samples=300, noise=0.05)

model = DBSCAN(eps=0.2, min_samples=5)
model.fit(X)

plt.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap='rainbow')
plt.title("DBSCAN from scratch")
plt.show()
