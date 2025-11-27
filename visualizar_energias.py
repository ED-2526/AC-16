import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from skimage.filters.rank import entropy
from skimage.morphology import disk

# --------------------------------------------
# 1) Función para reescalar
# --------------------------------------------
def reescalar(image, max_size=512):
    h, w = image.shape[:2]
    if h > w:
        scale = max_size / h
    else:
        scale = max_size / w
    new_w, new_h = int(w*scale), int(h*scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA), scale

# --------------------------------------------
# 2) Keypoints densos (sin SIFT)
# --------------------------------------------
def dense_kps(image, step=12, patch_size=16):
    image_res, _ = reescalar(image)
    gray = cv2.cvtColor(image_res, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    kps = [
        cv2.KeyPoint(float(x), float(y), float(patch_size))
        for y in range(patch_size, h - patch_size, step)
        for x in range(patch_size, w - patch_size, step)
    ]

    return image_res, gray, kps

# --------------------------------------------
# 3) Energía local = varianza del gradiente
# --------------------------------------------
def energia_patch(gray, kp, patch=16):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)

    x, y = int(kp.pt[0]), int(kp.pt[1])

    x1, x2 = max(0, x - patch), min(gray.shape[1], x + patch)
    y1, y2 = max(0, y - patch), min(gray.shape[0], y + patch)

    return np.var(mag[y1:y2, x1:x2])

# --------------------------------------------
# 4) CLUSTERIZAR patches por nivel de energía
# --------------------------------------------
def clusterizar_energia(image, out_dir="clusters_energia",
                        patch_size=16, step=12, K=4):
    """
    K = número de niveles de energía que quieres agrupar.
    """
    os.makedirs(out_dir, exist_ok=True)

    img_res, gray, kps = dense_kps(image, step=step, patch_size=patch_size)

    # ---- calcular energía ----
    energias = np.array([energia_patch(gray, kp, patch=patch_size) for kp in kps])

    # ---- normalizar ----
    energias_norm = energias.reshape(-1, 1)

    # ---- K-means ----
    km = KMeans(n_clusters=K, random_state=0)
    labels = km.fit_predict(energias_norm)

    # ---- crear carpetas ----
    for c in range(K):
        os.makedirs(os.path.join(out_dir, f"nivel_{c}"), exist_ok=True)

    # ---- guardar patches ----
    for kp, energia, cluster in zip(kps, energias, labels):
        x, y = int(kp.pt[0]), int(kp.pt[1])
        patch = img_res[y-patch_size:y+patch_size, x-patch_size:x+patch_size]

        fname = os.path.join(out_dir, f"nivel_{cluster}",
                             f"patch_{x}_{y}_{energia:.2f}.png")
        cv2.imwrite(fname, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))

    # ---- mostrar histograma ----
    plt.figure(figsize=(8,5))
    plt.hist(energias, bins=40, color="royalblue")
    plt.title("Histogramas de energías locales")
    plt.xlabel("Energía")
    plt.ylabel("Frecuencia")
    plt.grid(True)
    plt.show()

    print("✔ Parches generados y clusterizados en", out_dir)
    print("✔ Energías min/med/max:", energias.min(), energias.mean(), energias.max())

    return energias, labels, km

def kp_entropy_score(gray, kp, patch_size=12):
    x, y = int(kp.pt[0]), int(kp.pt[1])
    patch = gray[max(0,y-patch_size):y+patch_size,
                 max(0,x-patch_size):x+patch_size]

    if patch.shape[0] < 2*patch_size:
        return 0  # patch incompleto → descartar

    e = entropy(patch, disk(3)).mean()
    return float(e)

def clasificar_por_cuartiles(image, patch_size=16, step=12, out_dir="salida"):
    """
    Analiza una imagen, calcula la entropía por patch, divide por cuartiles
    y guarda cada patch en una carpeta correspondiente.
    """
    # ======== 1. Crear keypoints densos =========
    image_res, gray, kps = dense_kps(image, step=step, patch_size=patch_size)

    # ======== 2. Calcular entropía =========
    entropies = np.array([
        kp_entropy_score(gray, kp, patch_size=patch_size)
        for kp in kps
    ])

    # ======== 3. Calcular cuartiles =========
    q1 = np.percentile(entropies, 25)
    q2 = np.percentile(entropies, 50)
    q3 = np.percentile(entropies, 75)

    print(f"Q1={q1:.3f}, Q2={q2:.3f}, Q3={q3:.3f}")

    # ======== 4. Crear carpetas ==========
    dirs = {
        "bajo": os.path.join(out_dir, "bajo"),
        "medio_bajo": os.path.join(out_dir, "medio_bajo"),
        "medio_alto": os.path.join(out_dir, "medio_alto"),
        "alto": os.path.join(out_dir, "alto")
    }

    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # ======== 5. Guardar patches según cuartil ==========
    for i, (kp, e) in enumerate(zip(kps, entropies)):
        x, y = int(kp.pt[0]), int(kp.pt[1])
        patch = image_res[
            y - patch_size : y + patch_size,
            x - patch_size : x + patch_size
        ]

        # Clasificación por cuartiles
        if e < q1:
            cat = "bajo"
        elif e < q2:
            cat = "medio_bajo"
        elif e < q3:
            cat = "medio_alto"
        else:
            cat = "alto"

        cv2.imwrite(os.path.join(dirs[cat], f"patch_{i:05d}.png"), patch)

    print("✔ Patches clasificados y guardados por cuartiles.")
    return entropies, (q1, q2, q3)

img = cv2.imread("../../toy_dataset/6.jpg")

clasificar_por_cuartiles(img, out_dir="../../clusters_mi_imagen",
                    patch_size=16, step=12)