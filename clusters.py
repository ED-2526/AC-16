from herraientas import *
from gestor_descriptor import GestorDescriptor
import cv2
import os
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from PIL import Image
from sklearn.decomposition import PCA, IncrementalPCA
from time import time
import shutil
from sklearn.preprocessing import normalize

def hellinger_normalization(H):
    # H es un array (n_samples, n_bins)

    # 1. Normalizar L1
    H_l1 = normalize(H, norm='l1')

    # 2. Raíz cuadrada (power normalization)
    H_root = np.sqrt(H_l1)

    # 3. Normalizar L2 (opcional)
    H_final = normalize(H_root, norm='l2')

    return H_final

def show_image(path):
    try:
        img = Image.open(path)
    except:
        print("No pude abrir:", path)
        return
    
    plt.figure("Imagen seleccionada")
    plt.imshow(img)
    plt.axis("off")
    plt.show(block=False)

ruta_x = "../X_test.pkl"
ruta_y = "../y_test.pkl"

tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=0)
filenames = listar_archivos("../test")
histogramas = joblib.load(ruta_x)
histogramas = hellinger_normalization(histogramas)
labels = joblib.load(ruta_y)
unique_labels = np.unique(labels)
label_to_int = {label: i for i, label in enumerate(unique_labels)}
labels = np.array([label_to_int[l] for l in labels])

X_pca = PCA(n_components=50).fit_transform(histogramas)
X_tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, max_iter=5000).fit_transform(X_pca)


fig, ax = plt.subplots()
scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], s=12, c=labels, cmap="tab10")
ax.set_title("t-SNE (clic para ver imagen)")


# Guardamos las coordenadas y paths
points = X_tsne
paths = filenames


def on_click(event):
    # si no se clicó dentro del gráfico, ignoramos
    if event.inaxes != ax:
        return

    # coordenadas del clic
    x_click, y_click = event.xdata, event.ydata

    # distancia del clic a todos los puntos
    distances = np.sqrt((points[:, 0] - x_click)**2 + (points[:, 1] - y_click)**2)
    idx = np.argmin(distances)

    # si el clic está demasiado lejos de cualquier punto (>0.1), ignoramos
    if distances[idx] > 0.1:
        return

    print("Mostrando imagen:", paths[idx])
    estilo = os.path.split(os.path.split(paths[idx])[0])[1]
    print("Estilo:", estilo)
    show_image(paths[idx])


# conectar el evento
fig.canvas.mpl_connect("button_press_event", on_click)

plt.show()


