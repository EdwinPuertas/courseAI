from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

d = load_breast_cancer()
Xtr, Xte, ytr, yte = train_test_split(d.data, d.target, random_state=0)
rf = RandomForestClassifier(n_estimators=300, random_state=0).fit(Xtr, ytr)

imp = permutation_importance(rf, Xte, yte, n_repeats=20, random_state=0)
orden = imp.importances_mean.argsort()[::-1][:5]
print("Top-5 por importancia de permutación (caída en accuracy):")
for i in orden:
    m, s = imp.importances_mean[i], imp.importances_std[i]
    print(f"  {d.feature_names[i]:22s} {m:.4f} ± {s:.4f}")
