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
import shutil

y = joblib.load("../y_test.pkl")
y_pred = joblib.load("../y_pred.pkl")
ruta_real = "../test"
ruta_pkl = "../descriptores_test"
labels = listar_estilos(ruta_pkl)
output = "../result"
archivos = listar_archivos(ruta_pkl, terminacion=".pkl")
for true_y, pred, route in zip(y, y_pred, archivos):
    route = route.replace(ruta_pkl, ruta_real)
    route = route.replace(".pkl", ".jpg")
    destino = os.path.join(output, true_y, pred)
    os.makedirs(destino, exist_ok=True)
    shutil.copyfile(route, destino + "/" + os.path.basename(route))
