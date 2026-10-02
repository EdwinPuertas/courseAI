import numpy as np
# Modelo de recompensa (RLHF) con preferencias humanas: Bradley-Terry
# rasgos de respuesta: [correcta, cita_fuente, longitud_excesiva]
pares = [([1, 1, 0], [1, 0, 1]), ([1, 0, 0], [0, 0, 0]), ([1, 1, 1], [0, 1, 0]),
         ([0, 1, 0], [0, 0, 1]), ([1, 0, 0], [1, 0, 1]), ([1, 1, 0], [0, 1, 0])]
G = np.array([p for p, _ in pares], float); R = np.array([r for _, r in pares], float)
w = np.zeros(3); sig = lambda z: 1 / (1 + np.exp(-z))

for _ in range(2000):                        # max log σ(r(preferida) − r(rechazada))
    p = sig((G - R) @ w)
    w += 0.1 * ((1 - p)[:, None] * (G - R)).sum(0)
print("Pesos recompensa [correcta, cita, larga]:", w.round(2))

# DPO: misma idea sin modelo de recompensa explícito
lp_pol, lp_ref, beta = np.array([-4.0, -6.0]), np.array([-5.0, -5.0]), 0.1
margen = beta * ((lp_pol[0] - lp_ref[0]) - (lp_pol[1] - lp_ref[1]))
print("Pérdida DPO:", round(-np.log(sig(margen)), 4))
