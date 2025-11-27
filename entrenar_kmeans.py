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

def edge_density(gray, kp, patch_size=12, thr=10):
    x, y = int(kp.pt[0]), int(kp.pt[1])
    patch = gray[y-patch_size:y+patch_size, x-patch_size:x+patch_size]
    if patch.shape[0] < 2*patch_size: return 0
    
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1)
    mag = np.sqrt(gx*gx + gy*gy)

    return np.mean(mag > thr)

def ver_patch(image, keypoint, patch_size=32, filename= None):
    x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
    patch = image[
        max(0, y - patch_size): y + patch_size,
        max(0, x - patch_size): x + patch_size
    ]
    print(f"{filename}: entropy: {kp_entropy_score_fast(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), keypoint, patch_size=patch_size):.3f}")
    if filename:
        cv2.imwrite(filename, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
    return patch

def fast_entropy(gray_patch, bins=16):
    """
    Entropía rápida basada en histograma.
    Mucho más rápida que skimage.filters.rank.entropy.
    """
    hist = np.bincount(gray_patch.ravel(), minlength=256).astype(float)
    hist /= hist.sum() + 1e-8
    nz = hist[hist > 0]
    return -(nz * np.log2(nz)).sum()
def kp_entropy_score_fast(gray, kp, patch_size=12):
    x, y = int(kp.pt[0]), int(kp.pt[1])
    patch = gray[y - patch_size:y + patch_size,
                 x - patch_size:x + patch_size]

    if patch.shape[0] < 2*patch_size or patch.shape[1] < 2*patch_size:
        return 0.0

    return fast_entropy(patch)
def kp_entropy_score(gray, kp, patch_size=12):
    x, y = int(kp.pt[0]), int(kp.pt[1])
    patch = gray[max(0,y-patch_size):y+patch_size,
                 max(0,x-patch_size):x+patch_size]

    if patch.shape[0] < 2*patch_size:
        return 0  # patch incompleto → descartar

    e = entropy(patch, disk(3)).mean()
    return float(e)


def filtrar_gradiente(gray, kps, percentil=30):
    # blur para eliminar ruido del resize
    gray_blur = cv2.GaussianBlur(gray, (5,5), 10)

    gx = cv2.Sobel(gray_blur, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray_blur, cv2.CV_32F, 0, 1)
    mag = np.sqrt(gx * gx + gy * gy)

    threshold = np.percentile(mag, percentil)

    return [kp for kp in kps
            if mag[int(kp.pt[1]), int(kp.pt[0])] >= threshold]

def filtrar_norma(descriptors, kps, threshold=0.2):
    if descriptors is None or len(descriptors) == 0:
        return [], None
    desc_norm = np.linalg.norm(descriptors, axis=1)
    mask = desc_norm > threshold
    descriptors = descriptors[mask]
    kps = [kp for i, kp in enumerate(kps) if mask[i]]
    return kps, descriptors
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
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized_image, scale_factor


def dense_sift(image, step=24, patch_size=32):
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
    #keypoints = filtrar_textura(gray, keypoints, patch=patch_size, threshold=28.0)
    entropy = [kp_entropy_score_fast(gray, kp, patch_size=patch_size) for kp in keypoints]
    keypoints = [keypoints[i] for i in np.argsort(entropy)[int(len(keypoints)*0.9):]]
    keypoints = filtrar_gradiente(gray, keypoints, percentil=40)
    #scores = np.array([kp_entropy_score_fast(gray, kp, patch_size=patch_size) for kp in keypoints])
    #thr = np.percentile(scores, 99)
    #keypoints = [kp for kp, s in zip(keypoints, scores) if s >= thr]
    # Extraer descriptores
    keypoints, descriptors = sift.compute(gray, keypoints)
    keypoints, descriptors = filtrar_norma(descriptors, keypoints, threshold=0.2)
    if descriptors is None or len(descriptors) == 0:
        print("⚠ No se extrajeron descriptores válidos.")
        return keypoints, None
    descriptors = descriptors.astype(np.float32)
    descriptors += 1e-7
    descriptors /= descriptors.sum(axis=1, keepdims=True)
    descriptors = np.sqrt(descriptors)
    descriptors = normalize(descriptors, norm="l2")
    #for i, kp in enumerate(keypoints):
    #    ver_patch(image, kp, patch_size=patch_size, filename=f"../../patches_test/patch_{i}.png")
    return keypoints, descriptors


root = "../../toy_dataset"
df = cargar_labels()
features = []

def split_train_test(df, size = None, prop_test=0.1, random_state=42):
    size = min(size, len(df))
    if size is None:
        size = len(df)
    subset_df = df.sample(n=size, random_state=random_state)
    test_df = subset_df[int(size*(1-prop_test)):]
    train_df = subset_df[:int(size*(1-prop_test))]
    return train_df, test_df

def preprocess_cluster(desc, pca = None):  
    if pca is None:
        pca = PCA(n_components=64, whiten=True)
        desc = pca.fit_transform(desc)
        joblib.dump(pca, "pca_sift_64.pkl")
    else:
        desc = pca.transform(desc)
    desc = normalize(desc, norm="l2")
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
        score_train.append(kmeans.score(desc_train)/len(desc_train))
        score_test.append(kmeans.score(desc_test)/len(desc_test))
    plt.plot(Ks, score_train, marker='o', label='Train')
    plt.plot(Ks, score_test, marker='x', label='Test')
    plt.xlabel('Número de clusters K')
    plt.ylabel('Inercia')
    plt.title('Codo de KMeans')
    plt.show()
    return score_train, score_test


def dense_kps(image, step=24, patch_size=32):
    """Solo crea keypoints (sin SIFT), útil para análisis."""
    image, _ = reescalar(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    keypoints = []

    for y in range(patch_size, h - patch_size, step):
        for x in range(patch_size, w - patch_size, step):
            keypoints.append(cv2.KeyPoint(float(x), float(y), float(patch_size)))

    return image, gray, keypoints

def histograma_entropia(image, patch_size=32, step=24, show_patches=False):
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
    plt.figure(figsize=(10,5))
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
        fig, axs = plt.subplots(3, 4, figsize=(8,6))
        axs = axs.ravel()

        for i, idx in enumerate(top_idx):
            kp = kps[idx]
            x, y = int(kp.pt[0]), int(kp.pt[1])
            p = image_res[
                y-patch_size:y+patch_size,
                x-patch_size:x+patch_size
            ]
            axs[i].imshow(p, cmap="gray")
            axs[i].set_axis_off()
            axs[i].set_title(f"{entropies[idx]:.2f}")

        plt.suptitle("Patches con alta entropía")
        plt.tight_layout()
        plt.show()

    return entropies

if __name__ == "__main__":
    subset_size = 20000
    prop_test = 0.2
    descriptors_train, descriptors_test = features_para_kmeans(
        df, root,
        subset_size=subset_size,
        prop_test=prop_test,
        dir_pca= None
    )
    Ks = [64, 128, 256, 512, 1024]
    visualize_inertia(descriptors_train, descriptors_test, Ks)

