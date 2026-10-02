from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
herramientas = {"logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
                "bosque": RandomForestClassifier(random_state=0),
                "boosting": HistGradientBoostingClassifier(random_state=0)}

def planificador(memoria):          # aquí iría el LLM: decide la siguiente acción
    pendientes = [h for h in herramientas if h not in memoria]
    return pendientes[0] if pendientes else "FIN"

memoria = {}
while (accion := planificador(memoria)) != "FIN":           # ciclo agente
    memoria[accion] = cross_val_score(herramientas[accion], X, y, cv=5).mean()
    print(f"acción={accion:9s} observación: accuracy={memoria[accion]:.3f}")
print("Decisión final:", max(memoria, key=memoria.get))
