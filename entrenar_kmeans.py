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
from sklearn.cluster import KMeans
import hashlib
import psutil
import pandas as pd
#------------------- memmaping ------------------

def crear_memmap_para_descriptores(n_imagenes, max_desc_por_imagen=300, pca_dim=64):
    N_total = n_imagenes * max_desc_por_imagen
    mm = np.memmap(
        "descriptors.dat",
        dtype='float32',
        mode='w+',
        shape=(N_total, pca_dim)
    )
    return mm, N_total

def entrenar_pca(root, df, N_desc, n_components=64):
    descriptors = []
    samples = 0
    for img in tqdm(image_generator(root, df), desc="Extrayendo descriptores para PCA", total=N_desc//300):
        kp, desc = dense_sift(img, debug=False)
        if desc is not None:
            descriptors.append(desc)
            samples += len(desc)
        if samples >= N_desc:
            break
    print(f"Total descriptors collected for PCA: {samples}")
    pca = PCA(n_components=n_components, whiten=True)
    pca.fit(np.vstack(descriptors))
    joblib.dump(pca, "pca_sift_64.pkl")
    return pca

def escribir_descriptores_pca_en_memmap(df, root, pca, max_desc=300):
    n_imagenes = len(df)
    pca_dim = pca.n_components_

    mm, N_total = crear_memmap_para_descriptores(n_imagenes, max_desc, pca_dim)

    img_index = {}
    pointer = 0

    for i, img in enumerate(tqdm(image_generator(root, df), desc="PCA + memmap")):
        kp, desc = dense_sift(img, debug=False)
        if desc is None or desc.shape[0] == 0:
            img_index[i] = (pointer, pointer)
            continue

        # proyectar PCA
        desc_pca = pca.transform(desc).astype(np.float32)
        desc_pca = normalize(desc_pca, norm='l2')
        n = desc_pca.shape[0]

        # escribir en el memmap
        mm[pointer : pointer + n] = desc_pca

        img_index[i] = (pointer, pointer + n)
        pointer += n

    mm.flush()

    # Recortar el archivo si no se llenó por completo
    print(f"Total descriptores escritos: {pointer}")
    mm2 = np.memmap(
        "descriptores_pca.dat",
        dtype='float32',
        mode='w+',
        shape=(pointer, pca_dim)
    )

    joblib.dump(img_index, "index_descriptores_pca.pkl")
    mm.flush()
    return mm2, img_index
#------------------- profiler -------------------

def binning(values, max_points=10, agg="mean"):
    n = len(values)
    if n <= max_points:
        return values  # nada que reducir
    
    bin_size = int(np.ceil(n / max_points))
    values = np.array(values)

    if agg == "mean":
        return np.array([values[i:i+bin_size].mean() for i in range(0, n, bin_size)])
    elif agg == "median":
        return np.array([np.median(values[i:i+bin_size]) for i in range(0, n, bin_size)])
    elif agg == "min":
        return np.array([values[i:i+bin_size].min() for i in range(0, n, bin_size)])
    elif agg == "max":
        return np.array([values[i:i+bin_size].max() for i in range(0, n, bin_size)])
    else:
        raise ValueError("agg debe ser mean, median, min o max")

def extract_descriptors_con_medicion(df, root, debug=False):
    tiempos = []
    memorias = []

    desc_total = []

    proceso = psutil.Process(os.getpid())

    for img in tqdm(image_generator(root, df), desc="DenseSIFT con medición", total=len(df)):
        t0 = time()

        kp, desc = dense_sift(img, debug=debug)

        t1 = time()
        tiempos.append(t1 - t0)

        mem_actual = proceso.memory_info().rss / (1024 * 1024)   # MB
        memorias.append(mem_actual)

        if desc is not None:
            desc_total.append(desc)

    if len(desc_total) > 0:
        desc_total = np.vstack(desc_total)
    else:
        desc_total = np.empty((0, 128), dtype=np.float32)

    # Guardar métricas
    df_medidas = pd.DataFrame({
        "iter": np.arange(len(tiempos)),
        "tiempo": tiempos,
        "memoria_MB": memorias
    })
    df_medidas.to_csv("medicion_dense_sift.csv", index=False)

    return desc_total, tiempos, memorias

def graficar_mediciones_binned(csv_path=None, tiempos=None, memorias=None,
                               max_points=2000, agg="mean"):

    # cargar datos igual que antes...
    if csv_path is not None:
        df = pd.read_csv(csv_path)
        tiempos = df["tiempo"].values
        memorias = df["memoria_MB"].values
        iters = df["iter"].values
    else:
        iters = np.arange(len(tiempos))

    # aplicar binning
    tiempos_b = binning(tiempos, max_points=max_points, agg=agg)
    memorias_b = binning(memorias, max_points=max_points, agg=agg)

    # nuevo eje de iteraciones binned
    iters_b = np.linspace(iters.min(), iters.max(), len(tiempos_b))

    # --- gráficos ---
    plt.figure(figsize=(12,5))
    plt.plot(iters_b, tiempos_b)
    plt.title(f"Tiempo por iteración (binning={agg})")
    plt.xlabel("Iteración")
    plt.ylabel("Tiempo (s)")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12,5))
    plt.plot(iters_b, memorias_b)
    plt.title(f"Memoria (binning={agg})")
    plt.xlabel("Iteración")
    plt.ylabel("Memoria (MB)")
    plt.grid(True)
    plt.show()

# ------------------ helpers ------------------

def kp_id(kp):
    return (int(round(kp.pt[0])), int(round(kp.pt[1])), int(round(kp.size)))

def unique_image_id(image):
    h = hashlib.sha1(image.tobytes()).hexdigest()
    return h[:12] 

def extract_patch_from_image(img, x, y, patch_size):
    h, w = img.shape[:2]
    x1 = max(0, x - patch_size)
    x2 = min(w, x + patch_size)
    y1 = max(0, y - patch_size)
    y2 = min(h, y + patch_size)
    return img[y1:y2, x1:x2], (x1, x2, y1, y2)

def debug_dump(kps_removed, image, gray, patch_size, outdir):
    os.makedirs(outdir, exist_ok=True)
    for i, kp in enumerate(kps_removed):
        ver_patch(image, kp, patch_size, f"{outdir}/{i}.png")


# ------------------ filtres ------------------

def fast_entropy(gray_patch, bins=16):
    """
    Entropía rápida basada en histograma.
    """
    if gray_patch.size == 0:
        return 0.0
    # ensure 0-255 uint8
    patch = gray_patch.ravel().astype(np.int32)
    hist = np.bincount(patch, minlength=256).astype(float)
    hist_sum = hist.sum()
    if hist_sum == 0:
        return 0.0
    hist /= (hist_sum + 1e-12)
    nz = hist[hist > 0]
    return float(-(nz * np.log2(nz)).sum())


def kp_entropy_score_fast(gray, kp, patch_size=12):
    x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
    patch, _ = extract_patch_from_image(gray, x, y, patch_size)

    # require full-size patch to be comparable
    if patch.shape[0] != 2 * patch_size or patch.shape[1] != 2 * patch_size:
        return 0.0

    return fast_entropy(patch)


def kp_entropy_score(gray, kp, patch_size=12):
    x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
    patch, _ = extract_patch_from_image(gray, x, y, patch_size)

    if patch.shape[0] != 2 * patch_size or patch.shape[1] != 2 * patch_size:
        return 0.0

    e = entropy(patch, disk(3)).mean()
    return float(e)


def structured_score(gray, kp, patch_size=16):
    e = kp_entropy_score_fast(gray, kp, patch_size)
    x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
    p, _ = extract_patch_from_image(gray, x, y, patch_size)
    if p.shape[0] != 2 * patch_size or p.shape[1] != 2 * patch_size:
        return 0.0
    gx = cv2.Sobel(p, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(p, cv2.CV_32F, 0, 1, ksize=3)
    mean_g = float(np.mean(np.sqrt(gx * gx + gy * gy)))
    return mean_g / (1.0 + e)


def ver_patch(image, keypoint, patch_size=16, filename=None):
    x, y = int(round(keypoint.pt[0])), int(round(keypoint.pt[1]))
    patch, _ = extract_patch_from_image(image, x, y, patch_size)
    if filename:
        cv2.imwrite(filename, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
    return patch


def filtrar_gradiente(gray, kps, percentil=30):
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 1.0)

    gx = cv2.Sobel(gray_blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)

    threshold = np.percentile(mag, percentil)

    kept = []
    h, w = gray.shape
    for kp in kps:
        yy = int(round(kp.pt[1]))
        xx = int(round(kp.pt[0]))
        y1 = max(0, yy - 1); y2 = min(h, yy + 2)
        x1 = max(0, xx - 1); x2 = min(w, xx + 2)
        local = mag[y1:y2, x1:x2]
        val = float(local.mean()) if local.size > 0 else 0.0
        if val >= threshold:
            kept.append(kp)
    return kept


def filtrar_norma(descriptors, kps, threshold=0.2):
    if descriptors is None or len(descriptors) == 0:
        return [], None
    desc_norm = np.linalg.norm(descriptors, axis=1)
    mask = desc_norm > threshold
    descriptors = descriptors[mask]
    new_kps = [kp for i, kp in enumerate(kps) if mask[i]]
    return new_kps, descriptors


# ------------------ IO / basic utils ------------------

def image_generator(root, df, file_col="FILE"):
    for arx in df[file_col]:
        path = os.path.join(root, arx)
        if os.path.exists(path):
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
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized_image, scale_factor

#hellinger
def rootsift(descriptors):
    desc = descriptors.astype(np.float32)
    desc /= (desc.sum(axis=1, keepdims=True) + 1e-12)
    desc = np.sqrt(desc)
    desc /= (np.linalg.norm(desc, axis=1, keepdims=True) + 1e-12)
    return desc


# ------------------ dense-sift pipeline ------------------

def dense_sift(image, step=8, patch_size=16, debug=False):
    image, _ = reescalar(image)
    if len(image.shape) == 3: #de color a balnco y negro
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.shape[2] == 3 else image
    else:
        gray = image

    h, w = gray.shape

    # Crear keypoints en rejilla regular
    keypoints = []
    for y in range(patch_size, h - patch_size, step): #no queremos cojer puntos de los bordes
        for x in range(patch_size, w - patch_size, step):
            kp = cv2.KeyPoint(float(x), float(y), float(patch_size))
            keypoints.append(kp)

    sift = cv2.SIFT_create() #algoritme per descriure els pts
    
    # creem el nom de la img que anirà dins de cada carpeta (les carpetes son els filtres). Aquest nom serà una carpeta que contindrà tots els punts que no han passat el filtre ()
    if debug:
        n = unique_image_id(image)
        inicio = {kp_id(kp): kp for kp in keypoints}
    
    # 1) Primer filtre: aquí estem filtran pels que tenen gradient més gran que 40, es altres no els volem    
    keypoints = filtrar_gradiente(gray, keypoints, percentil=40) 

 # anem a guradar els punts que no han passat aquest primer filtre
    if debug:
        # 1) gradient
        n_key = {kp_id(kp): kp for kp in keypoints}
        os.makedirs(f"../../debug/1gradiente/{n}/", exist_ok=True)
        removed_ids = set(inicio.keys()) - set(n_key.keys())
        removed_kps = [inicio[k] for k in removed_ids]
        debug_dump(removed_kps, image, gray, patch_size, f"../../debug/1gradiente/{n}")

    # 2) Segon filtre: ordena energía
    entropy_vals = np.array([kp_entropy_score_fast(gray, kp, patch_size=patch_size) for kp in keypoints]) # fa un array amb l'energia que te cada punt en el mateix ordre.
    thr = np.percentile(entropy_vals, 75) #decidim el threshold
    if debug: #guarda els punts a la carpeta del filtre energia
        idx = np.argsort(entropy_vals)
        os.makedirs(f"../../debug/2energia/{n}/", exist_ok=True)
        for j, i in enumerate(idx):
            if entropy_vals[i] < thr:
                ver_patch(image, keypoints[i], patch_size=patch_size, filename=f"../../debug/2energia/{n}/{j}_{entropy_vals[i]:.2f}.png")
    keypoints = [kp for kp, s in zip(keypoints, entropy_vals) if s >= thr] #recorre els dos arrays alhora i els que passen el threshold els guarda a keypoints

    # 3) estructura
    score = np.array([structured_score(gray, kp, patch_size=patch_size) for kp in keypoints])
    thr = np.percentile(score, 50)
    if debug:
        idx = np.argsort(score)
        os.makedirs(f"../../debug/3estructura/{n}/", exist_ok=True)
        for j, i in enumerate(idx):
            if score[i] < thr:
                ver_patch(image, keypoints[i], patch_size=patch_size, filename=f"../../debug/3estructura/{n}/{j}_{score[i]:.2f}.png")
    keypoints = [kp for kp, s in zip(keypoints, score) if s >= thr]

    # 4) compute SIFT on survivors
    keypoints, descriptors = sift.compute(gray, keypoints)
    
    if debug:
        inicio3 = {kp_id(kp): kp for kp in keypoints}
        os.makedirs(f"../../debug/4norma/{n}/", exist_ok=True)
    keypoints, descriptors = filtrar_norma(descriptors, keypoints, threshold=0.2)
    if debug:
        removed_by_norm = [inicio3[k] for k in (set(inicio3.keys()) - {kp_id(kp) for kp in keypoints})]
        debug_dump(removed_by_norm, image, gray, patch_size, f"../../debug/4norma/{n}")

    if descriptors is None or len(descriptors) == 0:
        print("⚠ No se extrajeron descriptores válidos.")
        return keypoints, None

    # 6) rootSIFT
    descriptors = rootsift(descriptors)
    if debug:
        os.makedirs(f"../../debug/final/{n}/", exist_ok=True)
        for i, kp in enumerate(keypoints):
            ver_patch(image, kp, patch_size=patch_size, filename=f"../../debug/final/{n}/{i}.png")

    return keypoints, descriptors


# ------------------ resto del pipeline ------------------
def split_train_test(df, size=None, prop_test=0.1, random_state=42):
    size = min(size, len(df)) if size is not None else len(df)
    subset_df = df.sample(n=size, random_state=random_state)
    test_df = subset_df[int(size * (1 - prop_test)) :]
    train_df = subset_df[: int(size * (1 - prop_test))]
    return train_df, test_df


def preprocess_cluster(desc, pca=None):
    if pca is None:
        pca = PCA(n_components=64, whiten=True)
        desc = pca.fit_transform(desc)
        joblib.dump(pca, "pca_sift_64.pkl")
    else:
        desc = pca.transform(desc)
    desc = normalize(desc, norm="l2")
    return desc, pca


def extract_descriptors(df, root, debug=False):
    descriptors = []
    sizes = []
    for img in tqdm(image_generator(root, df), desc="Extracting descriptors", total=len(df)):
        kp, desc = dense_sift(img, debug=debug)
        if desc is not None:
            descriptors.append(desc)
            sizes.append(len(desc))
    descriptors = np.vstack(descriptors)
    return descriptors


def features_para_kmeans(df, root, subset_size=2000, prop_test=0.2, random_state=42, dir_pca=None, debug=False):
    train_df, test_df = split_train_test(
        df, size=subset_size, prop_test=prop_test, random_state=random_state
    )
    descriptors_train = extract_descriptors(train_df, root, debug=debug)
    pca = joblib.load(dir_pca) if dir_pca is not None and os.path.exists(dir_pca) else None
    pca = None
    descriptors_train, pca = preprocess_cluster(descriptors_train, pca)
    descriptors_test = extract_descriptors(test_df, root, debug=debug)
    descriptors_test, _ = preprocess_cluster(descriptors_test, pca)
    return descriptors_train, descriptors_test


def train_kmeans(descriptors, n_clusters=512, random_state=42):
    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        max_iter=300,
        n_init="auto",
        random_state=random_state,
    )
    kmeans.fit(descriptors)
    joblib.dump(kmeans, f"kmeans_bow_{n_clusters}.pkl")
    return kmeans


def visualize_inertia(desc_train, desc_test, Ks):
    score_train = []
    score_test = []
    for K in tqdm(Ks, desc="Entrenando modelos con distintas K"):
        kmeans = train_kmeans(desc_train, n_clusters=K)
        score_train.append(kmeans.score(desc_train) / len(desc_train))
        score_test.append(kmeans.score(desc_test) / len(desc_test))
    plt.plot(Ks, score_train, marker="o", label="Train")
    plt.plot(Ks, score_test, marker="x", label="Test")
    plt.xlabel("Número de clusters K")
    plt.ylabel("Inercia")
    plt.title("Codo de KMeans")
    plt.show()
    return score_train, score_test


def compute_vlad(descriptors, kmeans):
    if descriptors is None or len(descriptors) == 0:
        return None
    centers = kmeans.cluster_centers_
    K, D = centers.shape
    labels = kmeans.predict(descriptors)
    vlad = np.zeros((K, D), dtype=np.float32)
    for i, desc in enumerate(descriptors):
        c = labels[i]
        residual = desc - centers[c]
        vlad[c] += residual
    vlad = np.sign(vlad) * np.sqrt(np.abs(vlad) + 1e-12)
    vlad = vlad.reshape(-1)
    vlad /= (np.linalg.norm(vlad) + 1e-12)
    return vlad

def dense_kps(image, step=8, patch_size=16):
    """Solo crea keypoints (sin SIFT), útil para análisis."""
    image, _ = reescalar(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    h, w = gray.shape
    keypoints = []

    for y in range(patch_size, h - patch_size, step):
        for x in range(patch_size, w - patch_size, step):
            keypoints.append(cv2.KeyPoint(float(x), float(y), float(patch_size)))

    return image, gray, keypoints


def histograma_entropia(image, patch_size=16, step=8, show_patches=False):
    """
    Calcula y muestra el histograma de entropía de los keypoints de una imagen.
    """
    image_res, gray, kps = dense_kps(image, step=step, patch_size=patch_size)

    entropies = []
    for kp in kps:
        entropies.append(kp_entropy_score_fast(gray, kp, patch_size=patch_size))

    entropies = np.array(entropies)
    q1 = np.percentile(entropies, 25)
    q2 = np.percentile(entropies, 50)
    q3 = np.percentile(entropies, 75)
    n9 = np.percentile(entropies, 90)

    # ---- HISTOGRAMA ----
    plt.figure(figsize=(10, 5))
    plt.hist(entropies, bins=50, color="royalblue")
    plt.axvline(q1, color="orange", linestyle="--", linewidth=2, label=f"Q1 = {q1:.3f}")
    plt.axvline(q2, color="red", linestyle="--", linewidth=2, label=f"Q2 (mediana) = {q2:.3f}")
    plt.axvline(q3, color="green", linestyle="--", linewidth=2, label=f"Q3 = {q3:.3f}")
    plt.axvline(n9, color="purple", linestyle="--", linewidth=2, label=f"P90 = {n9:.3f}")


    plt.title("Histograma de entropía por keypoint")
    plt.xlabel("Entropía")
    plt.ylabel("Frecuencia")
    plt.grid(True)
    plt.show()

    print(f"✔ Keypoints analizados: {len(entropies)}")
    print(f"✔ Entropía mínima: {entropies.min():.3f}")
    print(f"✔ Entropía media:  {entropies.mean():.3f}")
    print(f"✔ Entropía máx:    {entropies.max():.3f}")

    # ---- OPCIONAL: visualizar los patches de alta entropía ----
    if show_patches:
        top_idx = np.argsort(entropies)[-12:]   # 12 patches de mayor entropía
        fig, axs = plt.subplots(3, 4, figsize=(8, 6))
        axs = axs.ravel()

        for i, idx in enumerate(top_idx):
            kp = kps[idx]
            x, y = int(kp.pt[0]), int(kp.pt[1])
            p = image_res[
                y - patch_size : y + patch_size,
                x - patch_size : x + patch_size
            ]
            axs[i].imshow(p, cmap="gray")
            axs[i].set_axis_off()
            axs[i].set_title(f"{entropies[idx]:.2f}")

        plt.suptitle("Patches con alta entropía")
        plt.tight_layout()
        plt.show()

    return entropies


if __name__ == "__main__":
    root = "../../toy_dataset"
    df = cargar_labels()
    pca = entrenar_pca(root, df, N_desc=200000, n_components=64)
    desc, index_img = escribir_descriptores_pca_en_memmap(df, root, pca, max_desc=300)
    #print(desc.shape)
    #kmeans = train_kmeans(desc, n_clusters=216)
    #joblib.dump(kmeans, "kmeans_bow_216_final.pkl")
    #kmeans = MiniBatchKMeans(
    #    n_clusters=216,
    #    init="k-means++",
    #    batch_size=10000)
    #joblib.dump(kmeans, "kmeans_bow_216_final_mini.pkl")
    #subset_size = 3000
    #prop_test = 0.2
    #descriptors_train, descriptors_test = features_para_kmeans(
    #    df, root,
    #    subset_size=subset_size,
    #    prop_test=prop_test,
    #    dir_pca= None
    #)
    #subset_size = 2000  # o lo que quieras para medir
    #df_train = df.sample(n=subset_size, random_state=42)
    #descs, tiempos, memorias = extract_descriptors_con_medicion(
    #    df_train, root, debug=False
    #)

    #graficar_mediciones_binned(csv_path="medicion_dense_sift.csv", max_points=150, agg="mean")
    #Ks = [64, 128, 256, 512, 1024]
    #visualize_inertia(descriptors_train, descriptors_test, Ks)
