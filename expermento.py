import joblib
import numpy as np
from exploracion_labels import cargar_labels
import cv2
import os
from tqdm import tqdm
from time import time
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from skimage.filters.rank import entropy
from skimage.morphology import disk

def filtrar_gradiente(gray, kps, percentil=30):
    gray_blur = cv2.GaussianBlur(gray, (5,5), 0)

    gx = cv2.Sobel(gray_blur, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray_blur, cv2.CV_32F, 0, 1)
    mag = np.sqrt(gx * gx + gy * gy)

    threshold = np.percentile(mag, percentil)

    return [kp for kp in kps
            if mag[int(kp.pt[1]), int(kp.pt[0])] >= threshold]
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
def dense_sift(image, step=12, patch_size=16, directory=None):
    image, _ = reescalar(image)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.shape[2] == 3 else image
    else:
        gray = image

    h, w = gray.shape
    # Crear keypoints en rejilla regular
    keypoints = []
    for y in range(patch_size, h - patch_size, step):
        for x in range(patch_size, w - patch_size, step):
            kp = cv2.KeyPoint(float(x), float(y), float(patch_size))
            keypoints.append(kp)

    # Crear extractor SIFT
    sift = cv2.SIFT_create()
    # Extraer descriptores
    keypoints = filtrar_gradiente(gray, keypoints, percentil=90)
    keypoints, descriptors = sift.compute(gray, keypoints)
    descriptors = descriptors.astype(np.float32)
    descriptors += 1e-7
    descriptors /= descriptors.sum(axis=1, keepdims=True)
    descriptors = np.sqrt(descriptors)
    descriptors = normalize(descriptors, norm="l2")
    for i, kp in enumerate(keypoints):
        ver_patch(image, kp, patch_size=patch_size, filename=directory+f"patch_{i}.png")
    return keypoints, descriptors

def ver_patch(image, keypoint, patch_size=16, filename= None):
    x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
    patch = image[
        max(0, y - patch_size): y + patch_size,
        max(0, x - patch_size): x + patch_size
    ]
    if filename:
        cv2.imwrite(filename, patch)
    return patch
directory = "../../patch_test_90/"
os.makedirs(directory, exist_ok=True)
file = "../../toy_dataset/349.jpg"
img = cv2.imread(file)
dense_sift(img, directory=directory)