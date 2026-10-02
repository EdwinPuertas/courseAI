from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score

X, y = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)

# Supervisado: aprende de pares (x, y)
clf = LogisticRegression(max_iter=500).fit(Xtr, ytr)
print("Supervisado  accuracy:", round(accuracy_score(yte, clf.predict(Xte)), 3))

# No supervisado: solo ve x, descubre grupos
km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
print("No supervis. ARI vs y:", round(adjusted_rand_score(y, km.labels_), 3))
