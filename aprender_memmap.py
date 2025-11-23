import numpy as np
from exploracion_labels import cargar_labels
import cv2
import os
from tqdm import tqdm
from time import time
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt

def image_generator(root, df, file_col="FILE"):
    for arx in df[file_col]:
        path = os.path.join(root, arx)
        if not os.path.exists(path):
            yield None
        else:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            yield img

import cv2

def reescalar(image, max_size=512):
    h, w = image.shape[:2]
    if h > w:
        scale_factor = max_size / float(h)
    else:
        scale_factor = max_size / float(w)
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized_image, scale_factor


def dense_sift(image, step=8, patch_size=16):
    image, _ = reescalar(image)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.shape[2] == 3 else image
    else:
        gray = image
    
    h, w = gray.shape

    # Crear keypoints en rejilla regular
    keypoints = []
    for y in range(0, h, step):
        for x in range(0, w, step):
            kp = cv2.KeyPoint(float(x), float(y), float(patch_size))
            keypoints.append(kp)

    # Crear extractor SIFT
    sift = cv2.SIFT_create()

    # Extraer descriptores
    keypoints, descriptors = sift.compute(gray, keypoints)

    return keypoints, descriptors
root = "../../toy_dataset"
df = cargar_labels()
features = []

subset_size = 2000
prop_test = 0.2
subset_df = df.sample(n=subset_size, random_state=42)
test_df = subset_df[int(subset_size*(1-prop_test)):]
subset_df = subset_df[:int(subset_size*(1-prop_test))]
sample_desc = []
test_desc = []

for img in tqdm(image_generator(root, subset_df)):
    kp, desc = dense_sift(img)
    if desc is not None:
        sample_desc.append(desc)
for img in tqdm(image_generator(root, test_df)):
    kp, desc = dense_sift(img)
    if desc is not None:
        test_desc.append(desc)
sample_desc = np.vstack(sample_desc)
test_desc = np.vstack(test_desc)
print(sample_desc.shape)
print(test_desc.shape)

Ks = [64,128,256,512,1024,2048]
inertias = []

for K in tqdm(Ks, desc= "Probando K"):
    kmeans = MiniBatchKMeans(n_clusters=K, batch_size=20000)
    kmeans.fit(sample_desc)
    inertias.append(kmeans.score(test_desc))


plt.figure(figsize=(10,5))
plt.plot(Ks, inertias, marker="o")
plt.title("Elbow para elegir K (BoW)")
plt.xlabel("Número de clusters (K)")
plt.ylabel("Inercia")
plt.grid(True)
plt.show()