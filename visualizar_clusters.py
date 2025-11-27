import os
import cv2
import numpy as np
from entrenar_kmeans import *

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

        # ---------------------------------------
        # 2) PCA + WHITENING — Obligatorio
        # ---------------------------------------
        descriptors_pca = pca.transform(descriptors_raw)
        descriptors_pca = normalize(descriptors_pca, norm="l2")
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
            try:
                cv2.imwrite(fname, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
            except Exception as e:
                print(f"Error saving patch {fname}: {e}")

            cluster_counters[cluster] += 1

    print("✔ Patches representativos guardados con PCA + Whitening + Normalization.")
if __name__ == "__main__":
    images = []
    kps = []
    descs = []
    root = "../../toy_dataset"
    df = cargar_labels()
    subset_size = 1000
    train_df, test_df = split_train_test(df, size = subset_size, prop_test=1, random_state=42)
    print(test_df.head())
    t = 0
    for img in tqdm(image_generator(root, test_df), desc="Dense SIFT para guardar patches", total=len(test_df)):
        kp, desc = dense_sift(img)
        images.append(img)
        #for k in kp:
        #    t += 1
        #    ver_patch(img, k, filename=f"patches/patch_{t}.png")
        
        #histograma_entropia(img, patch_size=32, step=24, show_patches=False)
        kps.append(kp)
        descs.append(desc)
    kmeans = joblib.load("kmeans_bow_64.pkl")
    save_representative_patches(
        images=images,
        keypoints_img=kps,
        descriptors_img=descs,
        pca=joblib.load("pca_sift_64.pkl"),
        kmeans=kmeans,      # tu MiniBatchKMeans ya entrenado
        out_dir="../../patches_amen_2",
        patch_size=16,
        n_samples= 1  # puedes poner 10, 20, 50…
    )
