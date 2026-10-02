import numpy as np
# AIMA cap. 19: ¿esperar mesa? (clientes: Ninguno/Algunos/Lleno)
clientes = ["N","A","L","A","L","A","N","A","L","L","N","L"]
espera   = [ 0,  1,  0,  1,  0,  1,  0,  1,  0,  0,  0,  1 ]

def H(y):                                     # entropía en bits
    p = np.bincount(y, minlength=2) / len(y)
    return -sum(pi * np.log2(pi) for pi in p if pi > 0)

y = np.array(espera); c = np.array(clientes)
resto = sum((c == v).mean() * H(y[c == v]) for v in set(clientes))
print("H(Espera)            =", round(H(y), 3))
print("Resto(Clientes)      =", round(resto, 3))
print("Ganancia(Clientes)   =", round(H(y) - resto, 3))
