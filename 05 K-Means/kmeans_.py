import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class KMeans:
    def __init__(self, max_iter, plot_progress):
        self.max_iter = max_iter
        self.plot_progress = plot_progress

    def fit(self, K, X):
        self.X = X
        self.k = K

        for iteration in range(self.max_iter):
            self.iter = iteration
            self.init_centroids = self.initial_centroids(self.k, self.X)
            self.cluster = self.assign_clusters(centroids=self.init_centroids, k=self.k, x=self.X)
            self.cluster_mean = self.compute_cluster_means(cluster=self.cluster, x=self.X, k=self.k)
            self.new_centroids = self.cluster_mean
            
            if self.plot_progress:
                self.plot()

            if np.sum([self.euclidean_distance(self.init_centroids[i], self.new_centroids[i]) for i in range(self.k)]) < 0.0001:
                print("Converged at iteration", iteration)
                break
        
        return self.cluster

    def initial_centroids(self, k, x):
        m_samples, n_features = x.shape
        random_sample_idxs = np.random.choice(m_samples, k, replace=False)
        centroids = [x[idx] for idx in random_sample_idxs]
        return centroids
    
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum(np.square(x1 - x2)))
    
    def assign_clusters(self, centroids, x, k):
        cluster = np.zeros(len(x))
        for i in range(x.shape[0]):
            distance = []
            for j in range(k):
                cal_dis = self.euclidean_distance(x[i], centroids[j])
                distance.append(cal_dis)
            cluster[i] = np.argmin(distance)
        return cluster
    
    def compute_cluster_means(self, cluster, x, k):
        new_centroids = np.array([x[cluster == i].mean(axis=0) for i in range(k)])
        return new_centroids
    
    def draw_line(self, p1, p2, style="red", linewidth=1):
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], style, linewidth=linewidth)

    def plot_data_points(self, X, idx):
        plt.scatter(X[:, 0], X[:, 1], c=idx)
        
    def plot_progress_kMeans(self, X, centroids, previous_centroids, idx, K, iteration):
        self.plot_data_points(X, idx)
        x_values = [point[0] for point in centroids]
        y_values = [point[1] for point in centroids]
        plt.scatter(x_values, y_values, marker='*', c='red', linewidths=3)
        
        cent = np.array(centroids)
        pre_cent = np.array(previous_centroids)
        for j in range(cent.shape[0]):
            self.draw_line(cent[j, :], pre_cent[j, :])
        
        plt.title("Iteration number %d" % iteration)
        plt.show()

    def plot(self):
        self.plot_progress_kMeans(self.X, centroids=self.new_centroids, previous_centroids=self.init_centroids, idx=self.cluster, K=self.k, iteration=self.iter)


if __name__ == "__main__":
    n_samples = 300
    n_features = 2
    centers = 3
    cluster_std = 1.0

    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, cluster_std=cluster_std)
    kmeans = KMeans(max_iter=10, plot_progress=True)
    clusters = kmeans.fit(K=3, X=X)

    # print(clusters)

