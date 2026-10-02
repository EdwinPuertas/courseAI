import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
rng = np.random.default_rng(1)
ruido = y.copy(); idx = rng.choice(len(y), 110, replace=False); ruido[idx] ^= 1

m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
p = cross_val_predict(m, X, ruido, cv=5, method="predict_proba")
sospechosos = np.where(p[np.arange(len(y)), ruido] < 0.2)[0]     # baja confianza
ok = np.setdiff1d(np.arange(len(y)), sospechosos)

aciertos = np.isin(sospechosos, idx).sum()
print("Sospechosas:", len(sospechosos), "| realmente erróneas:", aciertos)
print("CV con ruido :", cross_val_score(m, X, ruido, cv=5).mean().round(3))
print("CV depurado  :", cross_val_score(m, X[ok], ruido[ok], cv=5).mean().round(3))
