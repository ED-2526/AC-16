import numpy as np
from exploracion_labels import cargar_labels
import cv2
import os
from tqdm import tqdm
from time import time

def image_generator(root, df, file_col="FILE"):
    for arx in df[file_col]:
        path = os.path.join(root, arx)
        if not os.path.exists(path):
            yield None
        else:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            yield img

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


def dense_sift(image, step=12, patch_size=16):
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
a = time()
i = 0
import matplotlib.pyplot as plt

times_list = []
features = []   # versión original
size_list = []
for img in tqdm(image_generator(root, df), desc="Dense SIFT (lista)", total=2500):
    t0 = time()
    kp, desc = dense_sift(img)
    features = np.vstack([features, desc]) if len(features) > 0 else desc
    times_list.append(time() - t0)
    i += 1
    if i >= 2500:
        break
def smooth_mean(values, block=12):
    values = np.array(values)
    n = len(values) // block
    return values[:n*block].reshape(n, block).mean(axis=1)

plt.plot(smooth_mean(times_list), label="Lista (RAM)", alpha=0.8)
plt.show()
print(np.array(features).shape)
