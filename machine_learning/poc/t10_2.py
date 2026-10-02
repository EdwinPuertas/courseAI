import numpy as np
from sklearn.linear_model import SGDClassifier
rng = np.random.default_rng(0)
w_src = rng.normal(size=20); w_tgt = w_src + rng.normal(0, 0.4, 20)   # tareas afines

def datos(w, n):
    X = rng.normal(size=(n, 20)); return X, (X @ w > 0).astype(int)

Xs, ys = datos(w_src, 5000); Xt, yt = datos(w_tgt, 30); Xe, ye = datos(w_tgt, 2000)
pre = SGDClassifier(loss="log_loss", random_state=0).fit(Xs, ys)      # preentrenamiento
print("Preentrenado sin ajuste :", round(pre.score(Xe, ye), 3))
for _ in range(5): pre.partial_fit(Xt, yt)                            # fine-tuning
print("Preentrenado + 30 ejemplos:", round(pre.score(Xe, ye), 3))
cero = SGDClassifier(loss="log_loss", random_state=0).fit(Xt, yt)
print("Desde cero con 30 ejemplos:", round(cero.score(Xe, ye), 3))
