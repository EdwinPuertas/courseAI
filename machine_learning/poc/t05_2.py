import itertools
# Red bayesiana de AIMA: Robo, Sismo -> Alarma -> Juan, María
P_R, P_S = 0.001, 0.002
P_A = {(1, 1): .95, (1, 0): .94, (0, 1): .29, (0, 0): .001}
P_J, P_M = {1: .90, 0: .05}, {1: .70, 0: .01}

def conj(r, s, a, j=1, m=1):
    p = (P_R if r else 1 - P_R) * (P_S if s else 1 - P_S)
    p *= P_A[(r, s)] if a else 1 - P_A[(r, s)]
    return p * (P_J[a] if j else 1 - P_J[a]) * (P_M[a] if m else 1 - P_M[a])

# Inferencia por enumeración: P(Robo | Juan=1, María=1)
num = {r: sum(conj(r, s, a) for s, a in itertools.product([0, 1], repeat=2)) for r in (0, 1)}
print("P(Robo | j, m) =", round(num[1] / (num[0] + num[1]), 3))
