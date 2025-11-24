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

subset_size = 2000
prop_test = 0.2
subset_df = df.sample(n=subset_size, random_state=42)
test_df = subset_df[int(subset_size*(1-prop_test)):]
subset_df = subset_df[:int(subset_size*(1-prop_test))]
sample_desc = []
test_desc = []
"""
for img in tqdm(image_generator(root, subset_df)):
    kp, desc = dense_sift(img)
    if desc is not None:
        sample_desc.append(desc)
"""
"""
for img in tqdm(image_generator(root, test_df)):
    kp, desc = dense_sift(img)
    if desc is not None:
        test_desc.append(desc)
"""
"""
sample_desc = np.vstack(sample_desc)
sample_desc = normalize(sample_desc, norm="l2")
#test_desc = np.vstack(test_desc)
#test_desc = normalize(test_desc, norm="l2")
pca = PCA(n_components=32, whiten=True)
sample_desc = pca.fit_transform(sample_desc)
joblib.dump(pca, "pca_sift_32.pkl")
#test_desc = pca.transform(test_desc)
print(sample_desc.shape)
#print(test_desc.shape)
"""
"""
Ks = [64,128,256,512,1024,2048]
inertias = []

for K in tqdm(Ks, desc= "Probando K"):
    kmeans = MiniBatchKMeans(n_clusters=K, batch_size=20000)
    kmeans.fit(sample_desc)
    inertias.append(kmeans.score(test_desc))
    joblib.dump(kmeans, f"kmeans_bow_{K}.pkl")



plt.figure(figsize=(10,5))
plt.plot(Ks, inertias, marker="o")
plt.title("Elbow para elegir K (BoW)")
plt.xlabel("Número de clusters (K)")
plt.ylabel("Inercia")
plt.grid(True)
plt.show()

"""
"""
K = 512
kmeans = MiniBatchKMeans(n_clusters=K, batch_size=20000)
kmeans.fit(sample_desc)
print("✔ KMeans entrenado y guardado como 'kmeans_bow.pkl'")
"""
import os
import cv2
import numpy as np
"""
"""
kmeans = joblib.load("kmeans_bow_512.pkl")
print("✔ KMeans cargado desde kmeans_bow_512.pkl")
def save_representative_patches(
    images, keypoints_img, descriptors_img,
    pca, kmeans,
    out_dir="patches", patch_size=32, n_samples=20
):

    os.makedirs(out_dir, exist_ok=True)

    # Centros del clustering (ya en espacio PCA-whitened)
    centers = kmeans.cluster_centers_
    K = centers.shape[0]

    # Contador global de archivos por cluster
    cluster_counters = {c: 0 for c in range(K)}

    for img_idx, (image, keypoints, descriptors_raw) in enumerate(
        tqdm(zip(images, keypoints_img, descriptors_img),
             total=len(images), desc="Guardando representacion")
    ):

        # Si no hay descriptores, saltamos
        if descriptors_raw is None or len(descriptors_raw) == 0:
            continue

        # ---------------------------------------
        # 1) NORMALIZAR descriptores (L2) — Obligatorio
        # ---------------------------------------
        descriptors_norm = normalize(descriptors_raw, norm="l2")

        # ---------------------------------------
        # 2) PCA + WHITENING — Obligatorio
        # ---------------------------------------
        descriptors_pca = pca.transform(descriptors_norm)

        # ---------------------------------------
        # 3) Clustering usando descriptores PCA
        # ---------------------------------------
        labels = kmeans.predict(descriptors_pca)

        # ---------------------------------------
        # 4) Distancias al centroide en espacio PCA-whitened
        # ---------------------------------------
        distances = np.linalg.norm(descriptors_pca - centers[labels], axis=1)

        # ---------------------------------------
        # 5) Seleccionar top-n patches más representativos
        # ---------------------------------------
        top_idx = np.argsort(distances)[:n_samples]

        for idx in top_idx:
            kp = keypoints[idx]
            cluster = labels[idx]

            x, y = int(kp.pt[0]), int(kp.pt[1])

            patch = image[
                max(0, y - patch_size): y + patch_size,
                max(0, x - patch_size): x + patch_size
            ]

            cluster_dir = os.path.join(out_dir, f"cluster_{cluster}")
            os.makedirs(cluster_dir, exist_ok=True)

            fname = os.path.join(
                cluster_dir,
                f"img{img_idx}_patch{cluster_counters[cluster]}.png"
            )
            cv2.imwrite(fname, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))

            cluster_counters[cluster] += 1

    print("✔ Patches representativos guardados con PCA + Whitening + Normalization.")
# -------------------------------
# EJEMPLO DE USO
# -------------------------------

# Tomar una imagen de tu test_df

images = []
kps = []
descs = []
for img in tqdm(image_generator(root, subset_df), desc="Dense SIFT para guardar patches"):
    kp, desc = dense_sift(img)
    images.append(img)
    kps.append(kp)
    descs.append(desc)
save_representative_patches(
    images=images,
    keypoints_img=kps,
    descriptors_img=descs,
    pca=joblib.load("pca_sift_32.pkl"),
    kmeans=kmeans,      # tu MiniBatchKMeans ya entrenado
    out_dir="../../patches_norm_512_gigante",
    patch_size=16,
    n_samples=3      # puedes poner 10, 20, 50…
)

