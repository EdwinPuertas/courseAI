import numpy as np
frases = ["el rey gobierna el reino", "la reina gobierna el reino",
          "el rey vive en el castillo", "la reina vive en el castillo",
          "el perro come carne", "el gato come carne",
          "el perro duerme en la casa", "el gato duerme en la casa"]
stop = {"el", "la", "en"}                               # sin artículos
tok = [[w for w in f.split() if w not in stop] for f in frases]
V = sorted({w for f in tok for w in f})
ix = {w: i for i, w in enumerate(V)}; C = np.zeros((len(V), len(V)))
for f in tok:                                           # co-ocurrencia ventana ±2
    for i, w in enumerate(f):
        for j in range(max(0, i - 2), min(len(f), i + 3)):
            if i != j: C[ix[w], ix[f[j]]] += 1

U, s, _ = np.linalg.svd(np.log1p(C))                   # LSA: embeddings densos
E = U[:, :4] * s[:4]; E /= np.linalg.norm(E, axis=1, keepdims=True)
for w in ["rey", "perro", "castillo"]:
    sims = E @ E[ix[w]]
    top = [V[k] for k in np.argsort(-sims) if V[k] != w][:2]
    print(f"vecinos de '{w}':", top)
