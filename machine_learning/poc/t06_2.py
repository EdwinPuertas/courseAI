from sklearn.datasets import load_digits, make_moons
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score as ARI

X, _ = load_digits(return_X_y=True)                 # 64 dimensiones
pca = PCA(n_components=0.90).fit(X)
print("PCA: componentes para 90 % varianza:", pca.n_components_, "de 64")

Xm, ym = make_moons(n_samples=400, noise=0.06, random_state=0)
km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(Xm)
db = DBSCAN(eps=0.2, min_samples=5).fit(Xm)
print("Lunas  ARI K-Means:", round(ARI(ym, km.labels_), 2),
      "| ARI DBSCAN:", round(ARI(ym, db.labels_), 2))
