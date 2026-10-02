import numpy as np
rng = np.random.default_rng(0)
d, r_real = 512, 8
W0 = rng.normal(size=(d, d))                          # pesos preentrenados (congelados)
dW = rng.normal(size=(d, r_real)) @ rng.normal(size=(r_real, d)) * 0.01  # ajuste ideal

U, s, Vt = np.linalg.svd(dW)
print(f"Fine-tuning completo: {d*d:,} parámetros")
for r in [1, 4, 8, 16]:
    B, A = U[:, :r] * s[:r], Vt[:r]                   # LoRA: dW ≈ B @ A
    err = np.linalg.norm(dW - B @ A) / np.linalg.norm(dW)
    print(f"LoRA r={r:2d}: {2*d*r:6,} parámetros ({2*d*r/(d*d):.1%})  error={err:.3f}")
W_adaptado = W0 + B @ A                               # se fusiona al desplegar
