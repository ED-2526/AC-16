import os
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from exploracion_labels import cargar_labels
import joblib

def image_generator(root, df, file_col="FILE"):
    for arx in df[file_col]:
        path = os.path.join(root, arx)
        if not os.path.exists(path):
            yield None
        else:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            yield img
root = "../../toy_dataset"
df = cargar_labels()
X_arx = df[["FILE"]]
y = df[["CLEAN_DATE"]]
X_arx_train, y_train, X_arx_test, y_test = train_test_split(X_arx, y, test_size= 0.3, random_state=42)
print(X_arx_train.shape, y_train.shape)