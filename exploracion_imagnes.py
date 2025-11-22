import pandas as pd
from exploracion_labels import cargar_labels
import cv2
import os
import numpy as np

def image_generator(root, df, file_col="FILE"):
    for arx in df[file_col]:
        path = os.path.join(root, arx)
        if not os.path.exists(path):
            yield None
        else:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            yield img

df = cargar_labels()
print(df.head())
for img in image_generator("../../toy_dataset", df):
    print(img.shape)
