import numpy as np
from sklearn.linear_model import LogisticRegression
rng = np.random.default_rng(0); n = 4000
grupo = rng.integers(0, 2, n)                         # atributo sensible (A/B)
merito = rng.normal(0, 1, n)
proxy = merito + 0.8 * grupo + rng.normal(0, .5, n)   # variable que filtra el grupo
y = (merito + rng.normal(0, .5, n) > 0).astype(int)

p = LogisticRegression().fit(np.c_[proxy], y).predict(np.c_[proxy])
tasa = [p[grupo == g].mean() for g in (0, 1)]
tpr = [p[(grupo == g) & (y == 1)].mean() for g in (0, 1)]
print(f"Tasa de aprobación  A={tasa[0]:.2f}  B={tasa[1]:.2f}  "
      f"brecha (paridad demográfica)={abs(tasa[0]-tasa[1]):.2f}")
print(f"TPR                 A={tpr[0]:.2f}  B={tpr[1]:.2f}  "
      f"brecha (igualdad de oportunidad)={abs(tpr[0]-tpr[1]):.2f}")
