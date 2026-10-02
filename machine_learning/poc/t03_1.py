import numpy as np
rng = np.random.default_rng(0)
x = rng.uniform(500, 3000, 50)                    # pies cuadrados
y = 50 + 0.12 * x + rng.normal(0, 20, 50)         # precio (miles)

X = np.c_[np.ones_like(x), (x - x.mean()) / x.std()]   # sesgo + x normalizado
w_cerrada = np.linalg.solve(X.T @ X, X.T @ y)          # ecuación normal

w = np.zeros(2); alpha = 0.1
for _ in range(500):                                   # descenso de gradiente
    grad = X.T @ (X @ w - y) / len(y)
    w -= alpha * grad

print("Ecuación normal :", w_cerrada.round(2))
print("Gradiente desc. :", w.round(2))
print("MSE final       :", round(np.mean((X @ w - y) ** 2), 1))
