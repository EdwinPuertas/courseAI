import numpy as np
from scipy.stats import ks_2samp
rng = np.random.default_rng(0)
entreno = rng.normal(3.0, 1.0, 5000)          # ingreso en entrenamiento
prod_ok = rng.normal(3.0, 1.0, 1000)          # producción semana 1
prod_drift = rng.normal(3.6, 1.3, 1000)       # producción semana 8

def psi(ref, act, bins=10):                   # Population Stability Index
    cortes = np.quantile(ref, np.linspace(0, 1, bins + 1)); cortes[[0, -1]] = -np.inf, np.inf
    r = np.histogram(ref, cortes)[0] / len(ref) + 1e-6
    a = np.histogram(act, cortes)[0] / len(act) + 1e-6
    return np.sum((a - r) * np.log(a / r))

for nombre, prod in [("semana 1", prod_ok), ("semana 8", prod_drift)]:
    ks = ks_2samp(entreno, prod)
    alerta = "ALERTA" if psi(entreno, prod) > 0.2 else "ok"
    print(f"{nombre}: PSI={psi(entreno, prod):.3f}  KS p={ks.pvalue:.3g}  -> {alerta}")
