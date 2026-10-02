from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

X, y = load_breast_cancer(return_X_y=True)
modelos = {
    "Árbol (CART)":      DecisionTreeClassifier(random_state=0),
    "Random Forest":     RandomForestClassifier(n_estimators=300, random_state=0),
    "Gradient Boosting": HistGradientBoostingClassifier(random_state=0),
}
for nombre, m in modelos.items():
    s = cross_val_score(m, X, y, cv=5, scoring="roc_auc")
    print(f"{nombre:18s} AUC = {s.mean():.3f} ± {s.std():.3f}")
