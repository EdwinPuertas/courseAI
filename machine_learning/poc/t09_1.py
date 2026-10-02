import numpy as np
rng = np.random.default_rng(0)
tokens = ["el", "gato", "come", "pescado"]
d = 8; X = rng.normal(size=(4, d))                   # embeddings de entrada

Wq, Wk, Wv = (rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(3))
Q, K, V = X @ Wq, X @ Wk, X @ Wv

def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z)
    return e / e.sum(-1, keepdims=True)

A = softmax(Q @ K.T / np.sqrt(d))                   # Attention(Q,K,V)
Z = A @ V
print("Pesos de atención (filas suman 1):")
for t, fila in zip(tokens, A): print(f"{t:8s}", fila.round(2))
print("Salida:", Z.shape)
