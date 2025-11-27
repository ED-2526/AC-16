import pandas as pd
from exploracion_labels import cargar_labels
import cv2
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import joblib
import matplotlib.pyplot as plt
import random as rd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def get_metadata(path):
    try:
        with Image.open(path) as img:
            width, height = img.size
            mode = img.mode  # "RGB", "RGBA", "L", etc.
            fmt = img.format  # "JPEG", "PNG", etc.
        return width, height, mode, fmt
    except:
        return None, None, None, None


def leer_metadatos(root, df, file_col="FILE"):
    results = []
    for arx in tqdm(df[file_col], desc="leyendo metadatos"):
        path = os.path.join(root, arx)

        if not os.path.exists(path):
            results.append((arx, None, None, None, None, None))
            continue

        w, h, mode, fmt = get_metadata(path)
        size = w * h
        results.append((arx, w, h, size, mode, fmt))

    return pd.DataFrame(results, columns=["file", "width", "height", "size", "mode", "format"])

def image_generator(root, df, file_col="FILE"):
    for arx in df[file_col]:
        path = os.path.join(root, arx)
        if not os.path.exists(path):
            yield None
        else:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            yield img
def analisi_metadatos(df, df_meta):
    print(df_meta["mode"].value_counts())
    print(df_meta["format"].value_counts())
    plt.scatter(df_meta["width"], df_meta["height"], s=1)
    plt.xlabel("ancho")
    plt.ylabel("alto")
    plt.title("Distribución de tamaños de las imágenes")
    plt.show()
    
    plt.scatter(df_meta["width"] / df_meta["height"], df_meta["width"]*df_meta["height"], s=1)
    plt.xlabel("ratio (w/h)")
    plt.ylabel("area")
    plt.title("Aspect ratio vs tamaño")
    plt.show()
    
    plt.hist(df_meta["width"] * df_meta["height"])
    plt.xlabel('Tamaño en millones de pixeles')
    plt.title('Histograma tamaños')
    plt.show()
    print(df.describe())
    df_meta["size"] = df_meta["width"] * df_meta["height"]
    print(df_meta.describe())
    df_meta[["file", "width", "height", "size"]].to_csv("metadatos.csv", index=False)
df = cargar_labels()
print(df.head())
#for img in image_generator("../../toy_dataset", df):
#    print(img.shape)
#df_m = leer_metadatos("../../toy_dataset", df)
#joblib.dump(df_m, "metadades.pkl")
#print(df_m.head())
df_meta = joblib.load("metadades.pkl")
analisi_metadatos(df, df_meta)
"""
def load_small_image(path, size=(128, 128)):
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img = img.resize(size)
            return np.array(img)#, dtype=np.float32)
    except:
        return None
def color_histogram(img, bins=32):
    hist = []
    for ch in range(3):
        h, _ = np.histogram(img[:,:,ch], bins=bins, range=(0, 256))
        hist.append(h)
    return np.concatenate(hist)

def contrast(img):
    gray = np.mean(img, axis=2)  # luminancia simple
    return gray.std()

def extract_features(root, df, file_col="FILE"):
    features = []
    bad_files = []

    for arx in tqdm(df[file_col], desc="extrayendo features"):
        path = os.path.join(root, arx)
        img = load_small_image(path)

        if img is None:
            bad_files.append(arx)
            continue
        
        hist = color_histogram(img, bins=32)
        ctr = contrast(img)

        feat = np.concatenate([hist, [ctr]])
        features.append(feat)

    return np.array(features), bad_files

features, _ = extract_features("../../toy_dataset", df.iloc[np.random.randint(0, df.shape[0] - 1, 1000)])


X = features
# escalar
X_scaled = StandardScaler().fit_transform(X)

# opcional: reducir a 10–20 dimensiones
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=8, random_state=0)
labels = kmeans.fit_predict(X_pca)

pca2 = PCA(n_components=2)
XY = pca2.fit_transform(X_scaled)

plt.scatter(XY[:,0], XY[:,1], c=labels, s=2)
plt.title("Clusters por histograma + contraste")
plt.show()"""