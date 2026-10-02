import numpy as np
# Pasillo de 6 celdas: meta en la celda 5 (+1), cada paso cuesta -0.04
n, acciones = 6, [-1, +1]                        # izquierda, derecha
Q = np.zeros((n, 2)); rng = np.random.default_rng(0)
alpha, gamma, eps = 0.5, 0.9, 0.2

for episodio in range(300):
    s = 0
    while s != n - 1:
        a = rng.integers(2) if rng.random() < eps else Q[s].argmax()
        s2 = min(max(s + acciones[a], 0), n - 1)
        r = 1.0 if s2 == n - 1 else -0.04
        Q[s, a] += alpha * (r + gamma * Q[s2].max() - Q[s, a])   # Q-learning
        s = s2

print("Política:", "".join("←→"[a] for a in Q[:-1].argmax(1)), "META")
print("V(s):", Q.max(1).round(2))
