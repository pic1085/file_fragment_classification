# cluster_stage.py  ← 새 파일
import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class ClusterStage:
    def __init__(self, n_clusters=11, pca_components=50):
        self.n_clusters = n_clusters
        self.pca = PCA(n_components=pca_components, random_state=42)
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_to_category = {}  # 클러스터 ID → 카테고리 이름 매핑

    def fit(self, X, y_true_category_labels):
        Xs = self.scaler.fit_transform(X)
        Xr = self.pca.fit_transform(Xs)
        self.kmeans.fit(Xr)
        self._map_clusters(y_true_category_labels)

    def _map_clusters(self, y_cat):
        """각 클러스터에서 가장 많은 카테고리로 매핑 (순도 기반)"""
        from collections import Counter
        labels = self.kmeans.labels_
        for cid in range(self.n_clusters):
            mask = labels == cid
            most_common = Counter(y_cat[mask]).most_common(1)[0][0]
            self.cluster_to_category[cid] = most_common

    def predict(self, X):
        Xs = self.scaler.transform(X)
        Xr = self.pca.transform(Xs)
        cluster_ids = self.kmeans.predict(Xr)
        return np.array([self.cluster_to_category[c] for c in cluster_ids])

    def save(self, path):
        joblib.dump(self.__dict__, path, compress=3)

    def load(self, path):
        self.__dict__.update(joblib.load(path))