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
    
    plt.hist(df_meta["width"] * df_meta["height"], bins= 500)
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
print(df_meta.head())
analisi_metadatos(df, df_meta)
