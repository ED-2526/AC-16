from herraientas import *
from gestor_descriptor import GestorDescriptor
import cv2
import os
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
from tqdm import tqdm
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, Normalizer
from calc_descriptor import procesar_carpeta_imagenes
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import PCA, IncrementalPCA
from time import time

#============== Normalización ==============

def rootsift(descriptors, eps=1e-7):
    descriptors = descriptors / (descriptors.sum(axis=1, keepdims=True) + eps)
    descriptors = np.sqrt(descriptors)
    return descriptors

def l1(x, eps=1e-7):
    x = np.asarray(x, dtype=np.float64)

    # Caso vector 1D
    if x.ndim == 1:
        s = x.sum() + eps
        return x / s

    # Caso matriz 2D
    elif x.ndim == 2:
        s = x.sum(axis=1, keepdims=True) + eps
        return x / s

    else:
        raise ValueError("l1() solo acepta vectores 1D o matrices 2D")

def l2(x, eps=1e-7):
    x = np.asarray(x, dtype=np.float64)

    # Caso vector 1D
    if x.ndim == 1:
        n = np.linalg.norm(x) + eps
        return x / n

    # Caso matriz 2D
    elif x.ndim == 2:
        n = np.linalg.norm(x, axis=1, keepdims=True) + eps
        return x / n

    else:
        raise ValueError("l2() solo acepta vectores 1D o matrices 2D")

#==================== PCA ====================

#============== Entrenar Kmeans ==============


def entrenar_kmeans(train, n_clusters= 2000, batch_size=5000, max_iter=200, normalization=None, load = None, save = None):
    if load:
        return joblib.load(load)
    kmeans = MiniBatchKMeans(n_clusters= n_clusters, batch_size=batch_size, max_iter=max_iter)
    for desc in tqdm(train.cargar_todos(), desc="Entrenando kmeans", total=len(train)):
        if normalization == "l1":
            desc = l1(desc)
        elif normalization == "l2":
            desc = l2(desc)
        elif normalization.lower() == "root sift":
            desc = rootsift(desc)
        kmeans.partial_fit(desc)
    if save:
        joblib.dump(kmeans, save)
    return kmeans

#============== Encode ==============
def bow(gestor, kmeans, save_X = None, save_y = None, normalization= None, normalization_hist= None, power_norm=False):
    n_clusters = len(kmeans.cluster_centers_)
    X = np.zeros(shape=(len(gestor), n_clusters), dtype=np.float64)
    y = []
    estilos = gestor.get_estilos()
    i = 0
    for estilo in tqdm(estilos, desc="Cargando estilos", total=len(estilos)):
        for desc in tqdm(gestor.cargar_estilo(estilo), desc=f"Cargando estilo: {estilo}", total=gestor.len_estilo(estilo)):
            if normalization == "l1":
                desc = l1(desc)
            elif normalization == "l2":
                desc = l2(desc)
            elif normalization.lower() == "root sift":
                desc = rootsift(desc)
            hist = np.bincount(kmeans.predict(desc), minlength=n_clusters).astype(np.float64)
            if normalization_hist == "l1":
                hist = l1(hist)
            elif normalization_hist == "l2":
                hist = l2(hist)
            elif power_norm:
                hist = np.sqrt(hist + 1e-7)
                hist = l2(hist)
            X[i] = hist
            i += 1
            y.append(estilo)
    if save_X:
        joblib.dump(X, save_X)
    if save_y:
        joblib.dump(y, save_y)
    return X, y

def vlad(gestor, kmeans, save_X=None, save_y=None,
         normalization=None, power_norm=False, apply_pca=None, dim= 128):

    cluster_centers = kmeans.cluster_centers_
    n_clusters = len(cluster_centers)
    dim = dim

    X = np.zeros((len(gestor), n_clusters * dim), dtype=np.float64)
    y = []

    estilos = gestor.get_estilos()
    i = 0

    for estilo in tqdm(estilos, desc="Procesando estilos", total=len(estilos)):
        for desc in tqdm(gestor.cargar_estilo(estilo),
                         desc=f"VLAD {estilo}",
                         total=gestor.len_estilo(estilo)):

            # ============================
            # 1. Normalizar DESCRIPTORES
            # ============================
            if normalization == "l1":
                desc = l1(desc)
            elif normalization == "l2":
                desc = l2(desc)
            elif normalization is not None and normalization.lower() == "root sift":
                desc = rootsift(desc)

            labels = kmeans.predict(desc)
            vlad_vec = np.zeros((n_clusters, dim), dtype=np.float64)
            for k in range(n_clusters):
                # descriptores asignados al cluster k
                idx = np.where(labels == k)[0]
                if len(idx) > 0:
                    vlad_vec[k] = np.sum(desc[idx] - cluster_centers[k], axis=0)
            vlad_vec = vlad_vec.reshape(-1)
            if power_norm:
                vlad_vec = np.sign(vlad_vec) * np.sqrt(np.abs(vlad_vec) + 1e-7)
            vlad_vec = l2(vlad_vec)

            if apply_pca is not None:
                vlad_vec = apply_pca.transform([vlad_vec])[0]
                vlad_vec = l2(vlad_vec)
            X[i] = vlad_vec
            y.append(estilo)
            i += 1

    if save_X:
        joblib.dump(X, save_X)
    if save_y:
        joblib.dump(y, save_y)

    return X, y
if __name__ == "__main__":
    
    train_root = "../descriptores_train"
    test_root = "../descriptores_test"

    train = GestorDescriptor()
    test = GestorDescriptor()
    train.inicializar(train_root, existe=True)
    test.inicializar(test_root, existe=True)


    n_clusters= 512
    batch_size=5000
    max_iter = 200
    
    print("Entrenando kmeans...")
    kmeans = entrenar_kmeans(train, n_clusters=n_clusters, batch_size=batch_size, max_iter=max_iter,
                             normalization="root sift", load= "kmeans_512.pkl", save="kmeans_512.pkl")
    
    print("Obteniendo datos de train...")
    #X, y = vlad(train, kmeans, normalization="root sift", save_X="X_train_vlad.pkl", save_y="y_train_vlad.pkl",power_norm=True)
    #X, y = bow(train, kmeans, normalization="root sift", save_X="X_train.pkl", save_y="y_train.pkl", normalization_hist=None, power_norm=True)
    X = joblib.load("X_train_vlad.pkl")
    y = joblib.load("y_train_vlad.pkl")

    print("Entrenando modelo...")
    #model = SVC(kernel="lineal", probability=True)
    model = SVC(kernel="linear")
    scalar = StandardScaler()
    #X_norm = scalar.fit_transform(X)
    X_norm = X
    ini = time()
    model.fit(X_norm, y)
    joblib.dump(model, "model_rbf_norm_standard.pkl")
    print(time()-ini)

    print("Obteniendo datos de test...")
    X_test, y_test = vlad(test, kmeans, normalization="root sift", save_X="X_test_vlad.pkl", save_y="y_test_vlad.pkl", power_norm=True)
    #X_test, y_test = bow(test, kmeans, normalization="root sift", save_X="X_test.pkl", save_y="y_test.pkl", normalization_hist=None, power_norm=True)
    #X_test = joblib.load("X_test.pkl")
    #y_test = joblib.load("y_test.pkl")
    #X_norm_test = scalar.transform(X_test)
    X_norm_test = X_test
    print("Prediciendo resultados...")
    y_pred = model.predict(X_norm_test)
    joblib.dump(y_pred, "y_pred.pkl")

    print("Visualizando resultados...")
    cm = confusion_matrix(y_test, y_pred, labels=train.get_estilos())
    disp =ConfusionMatrixDisplay(cm, display_labels=train.get_estilos())
    disp.plot()
    plt.show()
    print(accuracy_score(y_test, y_pred))