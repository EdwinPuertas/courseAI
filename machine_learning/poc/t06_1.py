from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X, _ = make_blobs(n_samples=600, centers=4, cluster_std=1.0, random_state=7)

print(" k   inercia   silueta")
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    print(f"{k:2d} {km.inertia_:9.0f}   {silhouette_score(X, km.labels_):.3f}")
