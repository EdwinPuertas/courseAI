from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

for capas in [(8,), (64,), (128, 64)]:
    mlp = make_pipeline(StandardScaler(),
                        MLPClassifier(hidden_layer_sizes=capas, max_iter=2000,
                                      random_state=0))
    mlp.fit(Xtr, ytr)
    n_par = sum(w.size for w in mlp[-1].coefs_) + sum(b.size for b in mlp[-1].intercepts_)
    print(f"capas={str(capas):10s} parámetros={n_par:6d} accuracy={mlp.score(Xte, yte):.3f}")
