import numpy as np
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]); y = np.array([[0], [1], [1], [0]])  # XOR
rng = np.random.default_rng(0)
W1, b1 = rng.normal(0, 1, (2, 4)), np.zeros(4)
W2, b2 = rng.normal(0, 1, (4, 1)), np.zeros(1)
sig = lambda z: 1 / (1 + np.exp(-z))

for ep in range(5000):
    h = np.tanh(X @ W1 + b1); out = sig(h @ W2 + b2)      # forward
    d2 = out - y                                            # dL/dz (entropía cruzada)
    d1 = (d2 @ W2.T) * (1 - h ** 2)                         # backpropagation
    W2 -= 0.1 * h.T @ d2; b2 -= 0.1 * d2.sum(0)
    W1 -= 0.1 * X.T @ d1; b1 -= 0.1 * d1.sum(0)

print("Salida XOR:", out.ravel().round(3))
