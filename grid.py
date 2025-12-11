from herraientas import *
from gestor_descriptor import GestorDescriptor
import numpy as np
from tqdm import tqdm
import joblib
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from time import time
import matplotlib.pyplot as plt
import pandas as pd


# ================================
# Power norm (igual que tu código)
# ================================
def power_norm(x, eps=1e-7):
    x = np.sqrt(np.asarray(x, dtype=np.float64))
    n = np.linalg.norm(x, axis=1, keepdims=True) + eps
    return x / n


# ================================
# Entrenar KMeans
# ================================
def entrenar_kmeans(train, n_clusters=2000, batch_size=5000, max_iter=200):
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, max_iter=max_iter)

    for desc in tqdm(train.cargar_todos(), desc=f"KMeans K={n_clusters}", total=len(train)):
        kmeans.partial_fit(desc)

    return kmeans


# ================================
# BoW
# ================================
def bow(desc, kmeans):
    n_clusters = len(kmeans.cluster_centers_)
    return np.bincount(kmeans.predict(desc), minlength=n_clusters).astype(np.float64)


# ================================
# Procesar datos → BoW
# ================================
def procesar_datos(gestor, kmeans, n_clusters):
    X = np.zeros((len(gestor), n_clusters), dtype=np.float64)
    y = []

    estilos = gestor.get_estilos()
    i = 0

    for estilo in tqdm(estilos, desc="Estilos", total=len(estilos)):
        for desc in gestor.cargar_estilo(estilo):
            X[i] = bow(desc, kmeans)
            y.append(estilo)
            i += 1

    y = np.array(y)
    return X, y


# ================================
# GRID SEARCH + CSV
# ================================
def grid_search(train, test):

    K_values = [500, 1000, 2000]
    C_values = [1, 10, 100]
    gamma_values = ["scale", 1, 2, 3]

    mejor_acc = -1
    mejor_kmeans = None
    mejor_model = None
    mejor_setup = None

    # lista donde guardamos resultados
    registros = []

    for K in K_values:

        print(f"\n======================")
        print(f" Entrenando KMeans K={K}")
        print(f"======================")

        kmeans = entrenar_kmeans(train, n_clusters=K)

        X_train, y_train = procesar_datos(train, kmeans, K)
        X_test, y_test = procesar_datos(test, kmeans, K)

        X_train = power_norm(X_train)
        X_test = power_norm(X_test)

        clases = np.unique(y_train)

        for C in C_values:
            for gamma in gamma_values:

                print(f"\nSVM RBF: C={C}, gamma={gamma}")

                model = SVC(kernel="rbf", C=C, gamma=gamma)

                ini = time()
                model.fit(X_train, y_train)
                tiempo = time() - ini

                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                print(f"Accuracy={acc:.4f}  |  tiempo={tiempo:.2f}s")

                # ======== Métricas por clase ========
                report = classification_report(y_test, y_pred, output_dict=True)
                conf = confusion_matrix(y_test, y_pred, labels=clases)
                gamma = model._gamma
                fila = {
                    "K": K,
                    "C": C,
                    "gamma": gamma,
                    "accuracy": acc,
                    "tiempo": tiempo,
                    "precision_macro": report["macro avg"]["precision"],
                    "recall_macro": report["macro avg"]["recall"],
                    "f1_macro": report["macro avg"]["f1-score"]
                }

                # añadir métricas por clase
                for cls in clases:
                    fila[f"precision_{cls}"] = report[str(cls)]["precision"]
                    fila[f"recall_{cls}"] = report[str(cls)]["recall"]
                    fila[f"f1_{cls}"] = report[str(cls)]["f1-score"]

                # añadir confusiones entre clases
                for i, cls_i in enumerate(clases):
                    for j, cls_j in enumerate(clases):
                        fila[f"conf_{cls_i}_to_{cls_j}"] = conf[i, j]

                registros.append(fila)

                # ======= GUARDAR CSV EN CADA ITERACIÓN =======
                df = pd.DataFrame(registros)
                df.to_csv("resultados_grid.csv", index=False)
                print(" -> CSV actualizado")

                # actualizar mejor setup
                if acc > mejor_acc:
                    mejor_acc = acc
                    mejor_model = model
                    mejor_kmeans = kmeans
                    mejor_setup = (K, C, gamma)

    print("\n===================================")
    print("   MEJOR CONFIGURACIÓN ENCONTRADA")
    print("===================================")
    print(f"K = {mejor_setup[0]}")
    print(f"C = {mejor_setup[1]}")
    print(f"gamma = {mejor_setup[2]}")
    print(f"Accuracy = {mejor_acc:.4f}")

    return mejor_kmeans, mejor_model, mejor_setup


# ================================
# MAIN
# ================================
if __name__ == "__main__":

    train_root = "../descriptores_train"
    test_root = "../descriptores_test"

    train = GestorDescriptor()
    test = GestorDescriptor()

    train.inicializar(train_root, existe=True)
    test.inicializar(test_root, existe=True)

    train.set_norm("root sift")
    test.set_norm("root sift")

    kmeans, model, setup = grid_search(train, test)

    print("\nGuardando modelos...")
    joblib.dump(kmeans, "kmeans_mejor.pkl")
    joblib.dump(model, "svm_mejor.pkl")
