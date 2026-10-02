from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

banco = [("La entrega del taller se cerró antes de tiempo", "queja"),
         ("Excelente explicación de árboles de decisión", "elogio"),
         ("¿Cuándo es el parcial de IA?", "pregunta"),
         ("No funciona el enlace del notebook", "queja"),
         ("¿Puedo usar scikit-learn en el proyecto?", "pregunta")]
consulta = "El notebook no abre y la entrega vence hoy"

vec = TfidfVectorizer().fit([t for t, _ in banco] + [consulta])
sim = cosine_similarity(vec.transform([consulta]), vec.transform([t for t, _ in banco]))[0]
shots = [banco[i] for i in sim.argsort()[::-1][:3]]        # k = 3 más similares

prompt = "Clasifica el mensaje como queja, elogio o pregunta.\n\n"
prompt += "".join(f"Mensaje: {t}\nEtiqueta: {e}\n\n" for t, e in shots)
prompt += f"Mensaje: {consulta}\nEtiqueta:"
print(prompt)
