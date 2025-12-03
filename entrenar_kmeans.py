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
g = GestorDescriptor()

g.inicializar("../descriptores_train", existe=True)
class ModelKmeans:
    def __init__(self):
        self._model = None
        self._preproces = None

    def entrenar_kmeans(self, gestor, n_clusters=2000, batch_size=5000, max_iter=200, save = None):
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=batch_size,
            max_iter=max_iter,
        )
        if self._preproces is not None:
            for desc in tqdm(gestor.cargar_todos(), desc="Entrenando Kmeans", total=len(gestor)):
                desc = self._preproces.transform(desc)
                kmeans.partial_fit(desc)
        else:
            for desc in tqdm(gestor.cargar_todos(), desc="Entrenando Kmeans", total=len(gestor)):
                kmeans.partial_fit(desc)
        if save:
            joblib.dump(kmeans, save)
        self._model = kmeans

    def cargar_model(self, path):
        self._model = joblib.load(path)


def crear_bow(gestor, kmeans):



# entrenar kmeans:
n_clusters = 2000


        

#for desc in tqdm(g.cargar_todos(), desc="Entrenando Kmeans", total=len(g)):
#    kmeans.partial_fit(desc)
#
#joblib.dump(kmeans, "kmeans_provisional.pkl")
kmeans = joblib.load("kmeans_provisional.pkl")
"""n_art = len(os.listdir("../train/Art_Nouveau"))
n_exp = len(os.listdir("../train/Expressionism"))
N = n_art + n_exp
data = np.zeros((N, 2001), dtype=np.int64)
for i, desc in tqdm(enumerate(g.cargar_estilo("Art_Nouveau")), desc="Leyendo Art Nouveau", total = n_art):
    hist = np.bincount(kmeans.predict(desc), minlength=2000)
    data[i, :-1] = hist
    data[i,-1] = 0
print("============",i)
for i, desc in tqdm(enumerate(g.cargar_estilo("Expressionism"), start=n_art), desc="Leyendo Expresionismo", total=n_exp):
    hist = np.bincount(kmeans.predict(desc), minlength=2000)
    data[i, :-1] = hist
    data[i,-1] = 1
print("============", i)
joblib.dump(data, "matriz_prueba.pkl")"""
"""
data = joblib.load("matriz_prueba.pkl")
X = data[:,:-1]
y = data[:,-1]
print(X.shape, y.shape)
model = GradientBoostingClassifier()
model.fit(X, y)
joblib.dump(model, "modelo_juguete.pkl")"""
"""
model = joblib.load("modelo_juguete.pkl")
procesar_carpeta_imagenes("../prueba_test", "../descriptores_prueba", sobreescribir=False, existe_out=True)
g_test = GestorDescriptor()
g_test.inicializar("../descriptores_prueba", existe=True)
data = np.zeros((len(g_test), 2001), dtype=np.int64)
n_art = len(os.listdir("../prueba_test/Art_Nouveau"))
for i, desc in tqdm(enumerate(g_test.cargar_estilo("Art_Nouveau")), desc="Leyendo Art Nouveau"):
    hist = np.bincount(kmeans.predict(desc), minlength=2000)
    data[i, :-1] = hist
    data[i,-1] = 0
for i, desc in tqdm(enumerate(g_test.cargar_estilo("Expressionism"), start=n_art), desc="Leyendo Art Nouveau"):
    hist = np.bincount(kmeans.predict(desc), minlength=2000)
    data[i, :-1] = hist
    data[i,-1] = 1
"""
#joblib.dump(data, "prueba_test.pkl")
data = joblib.load("matriz_prueba.pkl")
tfidf = TfidfTransformer(norm=None) 
X = data[:,:-1]
y = data[:,-1]
print(X.shape, y.shape)
X = tfidf.fit_transform(X)

# 3. Normalización L2 (muy recomendada para SVM)
normalizer = Normalizer(norm='l2')
X = normalizer.fit_transform(X)
model = SVC(kernel="rbf")
model.fit(X, y)
joblib.dump(model, "modelo_juguete_svm_rbf.pkl")
data = joblib.load("prueba_test.pkl")
X_test = data[:, :-1]
X_test = tfidf.transform(X_test)
X_test = normalizer.transform(X_test)
y_test = data[:,-1]
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
