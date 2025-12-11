from collections import Counter, defaultdict
import joblib
import os

ruta_x = "../X_test.pkl"
ruta_y = "../y_test.pkl"

# Diccionario: cluster → lista de estilos
estilos_por_cluster = defaultdict(list)

for path_img, cluster in zip(filenames, labels):
    estilo = os.path.split(os.path.split(path_img)[0])[1]
    estilos_por_cluster[cluster].append(estilo)

print("\n===== Pureza de estilos por cluster =====\n")

for cluster, estilos in estilos_por_cluster.items():
    total = len(estilos)
    contador = Counter(estilos)
    estilo_principal, count_principal = contador.most_common(1)[0]
    pureza = count_principal / total * 100

    print(f"\nCluster {cluster}")
    print(f" Total imágenes: {total}")
    print(f" Estilo mayoritario: {estilo_principal} ({pureza:.2f}%)")
    print(" Distribución completa:")
    for est, cnt in contador.items():
        print(f"   {est}: {cnt} ({cnt/total*100:.1f}%)")