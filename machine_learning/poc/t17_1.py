from sklearn.datasets import load_wine
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

X, y = load_wine(return_X_y=True)
pipe = Pipeline([("esc", StandardScaler()), ("clf", LogisticRegression())])
espacio = [{"clf": [LogisticRegression(max_iter=2000)], "clf__C": [0.1, 1, 10]},
           {"clf": [SVC()], "clf__C": [1, 10], "clf__kernel": ["linear", "rbf"]},
           {"clf": [RandomForestClassifier(random_state=0)], "clf__n_estimators": [100, 300]}]
busq = GridSearchCV(pipe, espacio, cv=5).fit(X, y)              # mini-AutoML
print("Configuraciones evaluadas:", len(busq.cv_results_["params"]))
print("Mejor:", busq.best_params_["clf"].__class__.__name__,
      {k: v for k, v in busq.best_params_.items() if k != "clf"})
print("Accuracy CV:", round(busq.best_score_, 3))
