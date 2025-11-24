import os
import cv2
import numpy as np

def save_representative_patches(
    image, keypoints, descriptors, kmeans,
    out_dir="patches", patch_size=32, n_samples=20
):
    """
    Guarda los patches representativos por cluster.
    
    image: imagen original
    keypoints: lista de cv2.KeyPoint
    descriptors: matriz Nx128
    kmeans: modelo MiniBatchKMeans ya entrenado
    out_dir: carpeta base para guardar resultados
    patch_size: tamaño del patch alrededor del keypoint
    n_samples: número de patches representativos por cluster
    """

    # Crear carpetas
    os.makedirs(out_dir, exist_ok=True)
    K = kmeans.n_clusters
    centers = kmeans.cluster_centers_

    # Predecir cluster para cada descriptor
    labels = kmeans.predict(descriptors)

    # Agrupar (kp, desc) por cluster
    cluster_groups = {c: [] for c in range(K)}
    for kp, desc, lab in zip(keypoints, descriptors, labels):
        cluster_groups[lab].append((kp, desc))

    # Recorremos clusters
    for c in range(K):
        group = cluster_groups[c]
        if len(group) == 0:
            continue

        # Ordenar por distancia al centroide (más representativos primero)
        group_sorted = sorted(
            group,
            key=lambda x: np.linalg.norm(x[1] - centers[c])
        )

        # Crear carpeta del cluster
        cluster_dir = os.path.join(out_dir, f"cluster_{c}")
        os.makedirs(cluster_dir, exist_ok=True)

        # Guardar n_samples patches
        for i, (kp, desc) in enumerate(group_sorted[:n_samples]):
            x, y = int(kp.pt[0]), int(kp.pt[1])

            patch = image[
                max(0, y - patch_size) : y + patch_size,
                max(0, x - patch_size) : x + patch_size
            ]

            # Guardar archivo
            fname = os.path.join(cluster_dir, f"patch_{i}.png")
            cv2.imwrite(fname, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))

        print(f"Cluster {c}: guardados {min(n_samples, len(group))} patches")

    print("✔ Todos los patches representativos guardados.")



# -------------------------------
# EJEMPLO DE USO
# -------------------------------

# Tomar una imagen de tu test_df
sample_img = next(image_generator(root, test_df))
kp, desc = dense_sift(sample_img)

save_representative_patches(
    image=sample_img,
    keypoints=kp,
    descriptors=desc,
    kmeans=kmeans,      # tu MiniBatchKMeans ya entrenado
    out_dir="patches",
    patch_size=32,
    n_samples=15        # puedes poner 10, 20, 50…
)
