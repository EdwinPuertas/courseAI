import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

rng = np.random.default_rng(0); n = 400
df = pd.DataFrame({"edad": rng.normal(40, 12, n), "ingreso": rng.normal(3, 1, n),
                   "ciudad": rng.choice(["CTG", "BOG", "MED"], n)})
df.loc[rng.random(n) < 0.1, "ingreso"] = np.nan          # 10 % faltantes
y = ((df.edad > 38) & (df.ingreso.fillna(3) > 2.8)).astype(int)

prep = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("esc", StandardScaler())]), ["edad", "ingreso"]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["ciudad"])])
modelo = Pipeline([("prep", prep), ("clf", LogisticRegression())])
print("F1 CV (5 folds):", cross_val_score(modelo, df, y, cv=5, scoring="f1").mean().round(3))
