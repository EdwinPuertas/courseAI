from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

X, y = make_classification(n_samples=3000, weights=[0.95], random_state=0)   # 5 % positivos
Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, random_state=0)

for nombre, m in [("Dummy", DummyClassifier()), ("LogReg", LogisticRegression())]:
    m.fit(Xtr, ytr); p = m.predict(Xte); s = m.predict_proba(Xte)[:, 1]
    print(f"{nombre:6s} acc={accuracy_score(yte, p):.3f} F1={f1_score(yte, p):.3f} "
          f"AUC={roc_auc_score(yte, s):.3f}")
print("Matriz de confusión LogReg:\n", confusion_matrix(yte, p))
