import numpy as np
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

X, y = load_digits(return_X_y=True); rng = np.random.default_rng(0)
test = rng.choice(len(y), 500, replace=False)
pool = np.setdiff1d(np.arange(len(y)), test)

print("shots/clase  1-NN(sin entrenar)  LogReg(entrenada)")
for k in [1, 2, 4, 8, 16]:
    idx = np.concatenate([rng.choice(pool[y[pool] == c], k, replace=False)
                          for c in range(10)])
    knn = KNeighborsClassifier(1).fit(X[idx], y[idx])        # "memoriza" ejemplos
    lr = LogisticRegression(max_iter=2000).fit(X[idx], y[idx])
    a, b = knn.score(X[test], y[test]), lr.score(X[test], y[test])
    print(f"{k:6d}        {a:.3f}               {b:.3f}")
