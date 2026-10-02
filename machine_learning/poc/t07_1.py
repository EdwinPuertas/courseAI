from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

docs = ["el agente aprende una política con recompensas",
        "aprendizaje por refuerzo con recompensas y política",
        "la red convolucional clasifica imágenes",
        "clasificar imágenes con redes neuronales profundas"]

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(docs)
print("Vocabulario:", len(tfidf.vocabulary_), "términos | matriz", X.shape)
S = cosine_similarity(X)
for i in range(len(docs)):
    print(f"doc{i}", " ".join(f"{v:.2f}" for v in S[i]))
