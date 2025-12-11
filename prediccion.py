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

def power_norm(x, eps=1e-7):
    x = np.asarray(x, dtype=np.float64)
    x = np.sqrt(x)
    n = np.linalg.norm(x, axis=1, keepdims=True) + eps
    return x / n
#============== Entrenar Kmeans ==============


def entrenar_kmeans(train, n_clusters= 2000, batch_size=5000, max_iter=200, normalization=None, load = None, save = None):
    if load:
        return joblib.load(load)
    kmeans = MiniBatchKMeans(n_clusters= n_clusters, batch_size=batch_size, max_iter=max_iter)
    for desc in tqdm(train.cargar_todos(), desc="Entrenando kmeans", total=len(train)):
        kmeans.partial_fit(desc)
    if save:
        joblib.dump(kmeans, save)
    return kmeans

#============== Encode ==============
def bow(data, kmeans):
    n_clusters = len(kmeans.cluster_centers_)
    return np.bincount(kmeans.predict(data), minlength=n_clusters).astype(np.float64)

def procesar_datos(gestor, kmeans, save_X = None, save_y = None, normalization= None, normalization_hist= None, power_norm=False):
    X = np.zeros(shape=(len(gestor), n_clusters), dtype=np.float64)
    y = []
    estilos = gestor.get_estilos()
    i = 0
    for estilo in tqdm(estilos, desc="Cargando estilos", total=len(estilos)):
        for desc in tqdm(gestor.cargar_estilo(estilo), desc=f"Cargando estilo: {estilo}", total=gestor.len_estilo(estilo)):
            hist = bow(desc, kmeans)
            X[i] = hist
            i += 1
            y.append(estilo)
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
    train.set_norm("root sift")
    test.set_norm("root sift")


    n_clusters= 2000
    batch_size=5000
    max_iter = 200
    
    print("Entrenando kmeans...")
    kmeans = entrenar_kmeans(train, n_clusters=n_clusters, batch_size=batch_size, max_iter=max_iter, save="../kmeans_2000.pkl")
    
    print("Obteniendo datos de train...")
    #X, y = vlad(train, kmeans, normalization="root sift", save_X="X_train_vlad.pkl", save_y="y_train_vlad.pkl",power_norm=True)
    X, y = procesar_datos(train, kmeans, save_X="X_train.pkl", save_y="y_train.pkl")
    #X = joblib.load("../X_train.pkl")
    #y = joblib.load("../y_train.pkl")

    print("Entrenando modelo...")
    #model = SVC(kernel="lineal", probability=True)
    model = SVC(kernel="rbf")
    #X_norm = scalar.fit_transform(X)
    X_norm = power_norm(X)
    ini = time()
    model.fit(X_norm, y)
    joblib.dump(model, "../model_rbf.pkl")
    print(time()-ini)

    print("Obteniendo datos de test...")
    #X_test, y_test = vlad(test, kmeans, normalization="root sift", save_X="X_test_vlad.pkl", save_y="y_test_vlad.pkl", power_norm=True)
    X_test, y_test =  procesar_datos(test, kmeans, save_X="../X_test.pkl", save_y="../y_test.pkl")
    #X_test = joblib.load("../X_test.pkl")
    #y_test = joblib.load("../y_test.pkl")
    #X_norm_test = scalar.transform(X_test)
    X_norm_test = power_norm(X_test)
    X_norm_test = scalar.transform(X_norm_test)
    print("Prediciendo resultados...")
    y_pred = model.predict(X_norm_test)
    joblib.dump(y_pred, "../y_pred.pkl")
    print(accuracy_score(y_test, y_pred))
    print("Visualizando resultados...")
    cm = confusion_matrix(y_test, y_pred, labels=train.get_estilos())
    disp =ConfusionMatrixDisplay(cm, display_labels=train.get_estilos())
    disp.plot()
    plt.show()