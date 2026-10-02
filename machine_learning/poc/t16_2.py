import hashlib, json, time, joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

X, y = load_iris(return_X_y=True)
modelo = RandomForestClassifier(n_estimators=100, random_state=0).fit(X, y)
joblib.dump(modelo, "modelo_v1.joblib")

huella = hashlib.sha256(open("modelo_v1.joblib", "rb").read()).hexdigest()[:12]
tarjeta = {"nombre": "iris-rf", "version": "1.0.0", "sha256": huella}
json.dump(tarjeta, open("registro.json", "w"), indent=1)

cargado = joblib.load("modelo_v1.joblib")               # servicio de inferencia
t0 = time.perf_counter(); pred = cargado.predict(X[:1])
ms = (time.perf_counter() - t0) * 1e3
print("Registro:", tarjeta)
igual = bool((cargado.predict(X) == modelo.predict(X)).all())
print(f"Predicción={pred[0]}  latencia={ms:.1f} ms  consistente={igual}")
