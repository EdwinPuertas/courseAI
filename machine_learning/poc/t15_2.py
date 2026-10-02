import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X = StandardScaler().fit_transform(X)
Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0)
m = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
w = m.coef_[0]

for eps in [0.0, 0.1, 0.2, 0.4]:            # FGSM: x' = x + ε·sign(∇x pérdida)
    grad_sign = np.sign(np.outer(1 - 2 * yte, w))  # empuja hacia la clase contraria
    print(f"ε={eps:.1f}  accuracy={m.score(Xte + eps * grad_sign, yte):.3f}")
