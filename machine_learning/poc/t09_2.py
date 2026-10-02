import numpy as np
def pos_enc(n, d):                                    # codificación sinusoidal
    pos, i = np.arange(n)[:, None], np.arange(d)[None, :]
    ang = pos / 10000 ** (2 * (i // 2) / d)
    return np.where(i % 2 == 0, np.sin(ang), np.cos(ang))

n, d = 5, 8
rng = np.random.default_rng(1)
X = rng.normal(size=(n, d)) + pos_enc(n, d)
S = X @ X.T / np.sqrt(d)
S[np.triu_indices(n, k=1)] = -np.inf                 # máscara causal (GPT)
A = np.exp(S - S.max(1, keepdims=True)); A /= A.sum(1, keepdims=True)
print("Atención causal: cada token solo ve el pasado")
print(A.round(2))
