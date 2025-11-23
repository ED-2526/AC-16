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
a = time()
i = 0
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

times_list = []
features = []   # versión original
i = 0
N = len(df) * 4000
D = 128
k = 0
mm = np.memmap("dense_sift_desc.dat", dtype=np.float32, mode="w+", shape=(N, D))
for img in tqdm(image_generator(root, df), desc="Dense SIFT (lista)"):
    t0 = time.time()
    kp, desc = dense_sift(img)
    for p in desc:
        if p is not None:
            mm[k,:] = p
            k += 1
    times_list.append(time.time() - t0)
    if i >= 10000:
        break
    i += 1
def smooth_mean(values, block=20):
    values = np.array(values)
    n = len(values) // block
    return values[:n*block].reshape(n, block).mean(axis=1)

plt.plot(smooth_mean(times_list), label="Lista (RAM)", alpha=0.8)
plt.show()
print(np.array(features).shape)