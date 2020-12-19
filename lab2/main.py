from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from utils import read_data


def kmeans(X):
    kmeans = KMeans(n_clusters=3).fit(X)
    print("labels = ", kmeans.labels_)
    print("cluster centers = \n", kmeans.cluster_centers_)
    print("silhouette score = ", silhouette_score(X, kmeans.labels_))

    print(f"num of samples = {X.shape[0]}")
    for i in range(X.shape[0] - 3):
        kmeans = KMeans(n_clusters=3, init=X[i:i+3, :]).fit(X)
        print(f"{i}. silhouette score = {silhouette_score(X, kmeans.labels_)}")


if __name__ == '__main__':
    data = read_data()
    kmeans(data.to_numpy())
