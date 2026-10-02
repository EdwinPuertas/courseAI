import itertools, math
import numpy as np
# Modelo de crédito: f(ingreso, deuda, antigüedad); línea base = promedio
f = lambda x: 0.5 * x[0] - 0.8 * x[1] + 0.3 * x[0] * x[2]
x, base = np.array([4.0, 2.0, 1.0]), np.array([2.0, 1.0, 0.5])
nombres, n = ["ingreso", "deuda", "antigüedad"], 3

def v(S):                                   # valor de la coalición S
    z = base.copy(); z[list(S)] = x[list(S)]; return f(z)

phi = np.zeros(n)
for i in range(n):                          # valores de Shapley exactos
    otros = [j for j in range(n) if j != i]
    for k in range(n):
        for S in itertools.combinations(otros, k):
            w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
            phi[i] += w * (v(S + (i,)) - v(S))
for nm, p in zip(nombres, phi): print(f"{nm:11s} {p:+.3f}")
print("Suma =", round(phi.sum(), 3), "= f(x) - f(base) =", round(f(x) - f(base), 3))
