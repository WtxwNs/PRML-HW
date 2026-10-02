from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


def pairwise_squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    distances = (
        np.sum(left**2, axis=1, keepdims=True)
        + np.sum(right**2, axis=1)
        - 2.0 * left @ right.T
    )
    return np.maximum(distances, 0.0)


@dataclass
class KMeans:
    n_clusters: int = 10
    init: str = "kmeans++"
    n_init: int = 10
    max_iter: int = 300
    tol: float = 1e-4
    random_state: int = 42
    cluster_centers_: np.ndarray | None = None
    labels_: np.ndarray | None = None
    inertia_: float = np.inf
    n_iter_: int = 0

    def _initialize(self, features: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.init == "random":
            indices = rng.choice(len(features), self.n_clusters, replace=False)
            return features[indices].copy()
        if self.init != "kmeans++":
            raise ValueError(f"Unsupported initialization: {self.init}")

        centers = [features[rng.integers(len(features))]]
        closest = pairwise_squared_distances(features, np.asarray(centers)).ravel()
        for _ in range(1, self.n_clusters):
            total = closest.sum()
            index = rng.integers(len(features)) if total <= 0 else rng.choice(
                len(features), p=closest / total
            )
            centers.append(features[index])
            closest = np.minimum(
                closest,
                pairwise_squared_distances(features, features[index : index + 1]).ravel(),
            )
        return np.asarray(centers)

    def fit_predict(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] == 0 or not np.isfinite(features).all():
            raise ValueError("features must be a finite two-dimensional array with columns")
        for name in ("n_clusters", "n_init", "max_iter"):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if len(features) < self.n_clusters:
            raise ValueError("n_clusters cannot exceed the number of samples")

        # A fit is independent of previous datasets and their inertia scales.
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = np.inf
        self.n_iter_ = 0
        master_rng = np.random.default_rng(self.random_state)
        for _ in range(self.n_init):
            rng = np.random.default_rng(master_rng.integers(0, 2**32 - 1))
            centers = self._initialize(features, rng)
            labels = np.zeros(len(features), dtype=np.int64)

            for iteration in range(1, self.max_iter + 1):
                distances = pairwise_squared_distances(features, centers)
                labels = np.argmin(distances, axis=1)
                new_centers = centers.copy()
                for cluster in range(self.n_clusters):
                    members = features[labels == cluster]
                    if len(members):
                        new_centers[cluster] = members.mean(axis=0)
                    else:
                        new_centers[cluster] = features[rng.integers(len(features))]
                shift = np.linalg.norm(new_centers - centers)
                centers = new_centers
                if shift <= self.tol:
                    break

            distances = pairwise_squared_distances(features, centers)
            labels = np.argmin(distances, axis=1)
            inertia = distances[np.arange(len(features)), labels].sum()
            if inertia < self.inertia_:
                self.cluster_centers_ = centers
                self.labels_ = labels
                self.inertia_ = float(inertia)
                self.n_iter_ = iteration

        assert self.labels_ is not None
        return self.labels_.copy()


@dataclass
class DBSCAN:
    eps: float = 3.0
    min_samples: int = 5

    def fit_predict(self, features: np.ndarray) -> np.ndarray:
        distances = pairwise_squared_distances(features, features)
        neighborhoods = [np.flatnonzero(row <= self.eps**2) for row in distances]
        labels = np.full(len(features), -1, dtype=np.int64)
        visited = np.zeros(len(features), dtype=bool)
        cluster_id = 0

        for point in range(len(features)):
            if visited[point]:
                continue
            visited[point] = True
            neighbors = neighborhoods[point]
            if len(neighbors) < self.min_samples:
                continue

            labels[point] = cluster_id
            queue = list(neighbors[neighbors != point])
            queued = set(queue)
            cursor = 0
            while cursor < len(queue):
                neighbor = queue[cursor]
                cursor += 1
                if not visited[neighbor]:
                    visited[neighbor] = True
                    expanded = neighborhoods[neighbor]
                    if len(expanded) >= self.min_samples:
                        for candidate in expanded:
                            candidate = int(candidate)
                            if candidate not in queued:
                                queue.append(candidate)
                                queued.add(candidate)
                if labels[neighbor] == -1:
                    labels[neighbor] = cluster_id
            cluster_id += 1

        return labels


@dataclass
class LocalScaleSpectral:
    n_clusters: int = 10
    graph_neighbors: int = 12
    scale_neighbor: int = 3
    mutual: bool = True
    smoothing_neighbors: int = 5
    smoothing_steps: int = 3
    n_init: int = 30
    random_state: int = 42
    selection: str = "modularity"
    embedding_: np.ndarray | None = None
    labels_before_smoothing_: np.ndarray | None = None
    graph_: csr_matrix | None = None
    modularity_: float | None = None

    def _build_graph(self, features: np.ndarray) -> tuple[csr_matrix, np.ndarray]:
        if self.scale_neighbor > self.graph_neighbors:
            raise ValueError("scale_neighbor cannot exceed graph_neighbors")
        search = NearestNeighbors(n_neighbors=self.graph_neighbors + 1)
        distances, indices = search.fit(features).kneighbors(features)
        distances = distances[:, 1:]
        indices = indices[:, 1:]

        sample_count = len(features)
        rows = np.repeat(np.arange(sample_count), self.graph_neighbors)
        cols = indices.ravel()
        local_scale = distances[:, self.scale_neighbor - 1]
        denominator = np.maximum(local_scale[rows] * local_scale[cols], 1e-12)
        weights = np.exp(-(distances.ravel() ** 2) / denominator)
        directed = csr_matrix((weights, (rows, cols)), shape=(sample_count, sample_count))
        graph = directed.minimum(directed.T) if self.mutual else directed.maximum(directed.T)
        return graph, indices

    def _spectral_embedding(self, graph: csr_matrix) -> np.ndarray:
        degrees = np.asarray(graph.sum(axis=1)).ravel()
        inverse_sqrt = 1.0 / np.sqrt(np.maximum(degrees, 1e-12))
        laplacian = eye(graph.shape[0]) - diags(inverse_sqrt) @ graph @ diags(inverse_sqrt)
        initial_vector = np.linspace(1.0, 2.0, graph.shape[0])
        _, eigenvectors = eigsh(
            laplacian,
            k=self.n_clusters,
            which="SM",
            tol=1e-7,
            v0=initial_vector,
        )
        return normalize(eigenvectors)

    def _smooth_labels(self, labels: np.ndarray, neighbor_indices: np.ndarray) -> np.ndarray:
        if self.smoothing_steps <= 0 or self.smoothing_neighbors <= 0:
            return labels.copy()
        count = min(self.smoothing_neighbors, neighbor_indices.shape[1])
        smoothed = labels.copy()
        for _ in range(self.smoothing_steps):
            neighbor_labels = smoothed[neighbor_indices[:, :count]]
            votes = np.eye(self.n_clusters, dtype=np.int64)[neighbor_labels].sum(axis=1)
            smoothed = np.argmax(votes, axis=1)
        return smoothed

    @staticmethod
    def _modularity(graph: csr_matrix, labels: np.ndarray) -> float:
        degrees = np.asarray(graph.sum(axis=1)).ravel()
        total_weight = float(graph.sum())
        if total_weight <= 0:
            return float("-inf")
        score = 0.0
        for cluster in np.unique(labels):
            members = labels == cluster
            internal = float(graph[members][:, members].sum())
            volume = float(degrees[members].sum())
            score += internal / total_weight - (volume / total_weight) ** 2
        return score

    def fit_predict(self, features: np.ndarray) -> np.ndarray:
        graph, neighbor_indices = self._build_graph(features)
        embedding = self._spectral_embedding(graph)
        if self.selection != "modularity":
            raise ValueError(f"Unsupported restart selection criterion: {self.selection}")

        master_rng = np.random.default_rng(self.random_state)
        best_labels = None
        best_modularity = float("-inf")
        for _ in range(self.n_init):
            clusterer = KMeans(
                n_clusters=self.n_clusters,
                init="kmeans++",
                n_init=1,
                random_state=int(master_rng.integers(0, 2**32 - 1)),
            )
            labels = clusterer.fit_predict(embedding)
            labels = self._smooth_labels(labels, neighbor_indices)
            modularity = self._modularity(graph, labels)
            if modularity > best_modularity:
                best_labels = labels
                best_modularity = modularity

        assert best_labels is not None
        self.graph_ = graph
        self.embedding_ = embedding
        self.labels_before_smoothing_ = best_labels
        self.modularity_ = best_modularity
        return best_labels.copy()
