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

def split_train_test(df, size = None, prop_test=0.2, random_state=42):
    size = min(size, len(df))
    if size is None:
        size = len(df)
    subset_df = df.sample(n=size, random_state=random_state)
    test_df = subset_df[int(size*(1-prop_test)):]
    train_df = subset_df[:int(size*(1-prop_test))]
    return train_df, test_df

def preprocess_cluster(desc, pca = None):
    desc = normalize(desc, norm="l2")
    if pca is None:
        pca = PCA(n_components=64, whiten=True)
        desc = pca.fit_transform(desc)
        joblib.dump(pca, "pca_sift_64.pkl")
    else:
        desc = pca.transform(desc)
    return desc, pca

def extract_descriptors(df, root):
    descriptors = []
    for img in tqdm(image_generator(root, df), desc="Extracting descriptors", total=len(df)):
        kp, desc = dense_sift(img)
        if desc is not None:
            descriptors.append(desc)
    descriptors = np.vstack(descriptors)
    return descriptors

def features_para_kmeans(df, root, subset_size=2000, prop_test=0.2, random_state=42, dir_pca = None):
    train_df, test_df = split_train_test(df, size=subset_size, prop_test=prop_test, random_state=random_state)
    descriptors_train = extract_descriptors(train_df, root)
    pca = joblib.load(dir_pca) if dir_pca is not None  and os.path.exists(dir_pca) else None
    descriptors_train, pca = preprocess_cluster(descriptors_train, pca)
    descriptors_test = extract_descriptors(test_df, root)
    descriptors_test, _ = preprocess_cluster(descriptors_test, pca)
    return descriptors_train, descriptors_test

def train_kmeans(descriptors, n_clusters=512, batch_size=1000, random_state=42):
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        random_state=random_state
    )
    kmeans.fit(descriptors)
    joblib.dump(kmeans, f"kmeans_bow_{n_clusters}.pkl")
    return kmeans

def visualize_inertia(desc_train, desc_test, Ks):
    score_train = []
    score_test = []
    for K in tqdm(Ks, desc="Entrenando modelos con distintas K"):
        kmeans = train_kmeans(desc_train, n_clusters=K)
        score_train.append(kmeans.score(desc_train))
        score_test.append(kmeans.score(desc_test))
    plt.plot(Ks, score_train, marker='o', label='Train')
    plt.plot(Ks, score_test, marker='x', label='Test')
    plt.xlabel('Número de clusters K')
    plt.ylabel('Inercia')
    plt.title('Codo de KMeans')
    plt.show()
    return score_train, score_test
if __name__ == "__main__":
    subset_size = 2000
    prop_test = 0.2
    descriptors_train, descriptors_test = features_para_kmeans(
        df, root,
        subset_size=subset_size,
        prop_test=prop_test,
        dir_pca="pca_sift_64.pkl"
    )
    Ks = [64, 128, 256, 512, 1024]
    visualize_inertia(descriptors_train, descriptors_test, Ks)

