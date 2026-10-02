import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 5000)); y = rng.integers(0, 2, 100)   # ¡puro ruido!

# MAL: seleccionar variables con TODOS los datos y luego validar
Xsel = SelectKBest(f_classif, k=20).fit_transform(X, y)
fuga = cross_val_score(LogisticRegression(), Xsel, y, cv=5).mean()
print("Con fuga de datos :", round(fuga, 2))

# BIEN: la selección vive dentro del pipeline (se ajusta por fold)
pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression())
print("Sin fuga de datos :", cross_val_score(pipe, X, y, cv=5).mean().round(2))
