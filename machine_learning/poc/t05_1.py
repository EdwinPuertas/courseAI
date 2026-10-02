from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

textos = ["gana dinero gratis ahora", "oferta gratis solo hoy", "premio dinero rápido",
          "reunión del curso mañana", "entrega del taller de IA", "notas del parcial"]
y = ["spam", "spam", "spam", "ok", "ok", "ok"]

vec = CountVectorizer()
nb = MultinomialNB(alpha=1.0).fit(vec.fit_transform(textos), y)   # Laplace

nuevos = ["dinero gratis del curso", "taller de IA mañana"]
for t, p in zip(nuevos, nb.predict_proba(vec.transform(nuevos))):
    print(f"{t:26s} -> P(spam)={p[list(nb.classes_).index('spam')]:.2f}")
