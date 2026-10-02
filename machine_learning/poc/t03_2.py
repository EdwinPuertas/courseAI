import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X, y = load_diabetes(return_X_y=True)
for nombre, reg in [("OLS", LinearRegression()), ("Ridge", Ridge(alpha=10)),
                    ("Lasso", Lasso(alpha=5))]:
    m = make_pipeline(StandardScaler(), reg).fit(X, y)
    coef = m[-1].coef_
    r2 = cross_val_score(make_pipeline(StandardScaler(), reg), X, y, cv=5).mean()
    ceros = np.sum(np.isclose(coef, 0))
    print(f"{nombre:5s} R2={r2:.3f}  |w|={np.abs(coef).sum():6.1f}  ceros={ceros}")
