from collections import Counter, defaultdict

corpus = ("el modelo aprende de datos . el modelo predice la palabra . "
          "la red aprende de datos sin etiquetas .").split()

# Autosupervisado: las etiquetas salen del propio texto
pares = [(corpus[i], corpus[i + 1]) for i in range(len(corpus) - 1)]
print("Pares (x -> y) generados:", pares[:3])

conteo = defaultdict(Counter)
for x, y in pares:
    conteo[x][y] += 1

for w in ["el", "aprende", "de"]:
    print(f"P(siguiente | '{w}') ->", conteo[w].most_common(1)[0][0])
