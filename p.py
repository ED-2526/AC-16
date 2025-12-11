import os
import numpy as np
import joblib
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from gestor_descriptor import *


def hellinger_normalization(H):
    H_l1 = normalize(H, norm='l1')
    H_root = np.sqrt(H_l1)
    H_final = normalize(H_root, norm='l2')
    return H_final


def obtener_estilo(path):
    # estilo = carpeta padre del padre
    return os.path.split(os.path.split(path)[0])[1]


def pureza_promedio(filenames, labels):
    estilos_por_cluster = defaultdict(list)

    for path_img, cluster in zip(filenames, labels):
        estilo = obtener_estilo(path_img)
        estilos_por_cluster[cluster].append(estilo)

    purezas = []

    for cluster, estilos in estilos_por_cluster.items():
        total = len(estilos)
        if total == 0:
            continue
        contador = Counter(estilos)
        estilo_principal, count_principal = contador.most_common(1)[0]
        pureza = count_principal / total
        purezas.append(pureza)

    if len(purezas) == 0:
        return 0

    return np.mean(purezas)



# --------------------------
# Cargar TRAIN y TEST
# --------------------------

ruta_x_train = "../X_train.pkl"
ruta_y_train = "../y_train.pkl"
ruta_imgs_train = "../train"

ruta_x_test = "../X_test.pkl"
ruta_y_test = "../y_test.pkl"
ruta_imgs_test = "../test"

filenames_train = listar_archivos(ruta_imgs_train)
filenames_test  = listar_archivos(ruta_imgs_test)

X_train = hellinger_normalization(joblib.load(ruta_x_train))
X_test  = hellinger_normalization(joblib.load(ruta_x_test))



# ---------------------------------
# K values para la gráfica
# ---------------------------------

lista_K = [5, 10, 20, 40, 60, 80, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550]

purezas_train = []
purezas_test = []


print("Calculando purezas para diferentes K...\n")

for K in lista_K:
    print(f"--> K = {K}")

    kmeans = KMeans(n_clusters=K, random_state=0)

    # Entrenar solo con TRAIN
    labels_train = kmeans.fit_predict(X_train)

    # Usar los mismos centroides para TEST
    labels_test = kmeans.predict(X_test)

    # Calcular pureza promedio
    pureza_tr = pureza_promedio(filenames_train, labels_train)
    pureza_te = pureza_promedio(filenames_test,  labels_test)

    purezas_train.append(pureza_tr)
    purezas_test.append(pureza_te)

    print(f"   Pureza TRAIN: {pureza_tr:.4f}")
    print(f"   Pureza TEST:  {pureza_te:.4f}\n")


# ---------------------------------
# Graficar curvas
# ---------------------------------

plt.figure(figsize=(10,5))
plt.plot(lista_K, purezas_train, marker="o", label="Train")
plt.plot(lista_K, purezas_test, marker="o", label="Test")

plt.title("Relación entre K y pureza promedio")
plt.xlabel("K (número de clusters)")
plt.ylabel("Pureza promedio")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

