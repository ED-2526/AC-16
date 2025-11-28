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
from sklearn.decomposition import IncrementalPCA
from skimage.filters.rank import entropy
from skimage.morphology import disk
from sklearn.cluster import KMeans
import hashlib
import psutil
import pandas as pd
from scipy.sparse import lil_matrix
import gc

# ==================== CONFIGURACIÓN ====================
class Config:
    MAX_IMAGE_SIZE = 512
    DENSE_SIFT_STEP = 8
    DENSE_SIFT_PATCH_SIZE = 16
    PCA_COMPONENTS = 64
    CHUNK_SIZE = 1000  # Procesar imágenes en chunks
    DESCRIPTORS_PER_CHUNK = 50000  # Máximo descriptores por chunk
    N_CLUSTERS = 512
    RANDOM_STATE = 42

config = Config()

# ==================== CACHÉ DE DESCRIPTORES ====================
class DescriptorCache:
    def __init__(self, cache_dir="descriptor_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_key(self, df_subset):
        """Genera clave única para el subset de datos"""
        content = "".join(sorted(df_subset["FILE"].astype(str))) + f"_{len(df_subset)}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def save_descriptors(self, descriptors, cache_key):
        """Guarda descriptores en cache"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.npy")
        np.save(cache_file, descriptors)
        return cache_file
    
    def load_descriptors(self, cache_key):
        """Carga descriptores desde cache"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.npy")
        if os.path.exists(cache_file):
            return np.load(cache_file)
        return None

descriptor_cache = DescriptorCache()

# ==================== FUNCIONES OPTIMIZADAS ====================

def extract_descriptors_optimized(df, root, debug=False, use_cache=True):
    """
    Versión optimizada de extracción de descriptores con chunks y cache
    """
    if use_cache:
        cache_key = descriptor_cache.get_cache_key(df)
        cached = descriptor_cache.load_descriptors(cache_key)
        if cached is not None:
            print("✅ Descriptores cargados desde cache")
            return cached, [], []
    
    tiempos = []
    memorias = []
    all_descriptors = []
    
    proceso = psutil.Process(os.getpid())
    
    # Procesar por chunks para evitar sobrecarga de memoria
    n_chunks = (len(df) + config.CHUNK_SIZE - 1) // config.CHUNK_SIZE
    
    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * config.CHUNK_SIZE
        end_idx = min((chunk_idx + 1) * config.CHUNK_SIZE, len(df))
        chunk_df = df.iloc[start_idx:end_idx]
        
        print(f"📦 Procesando chunk {chunk_idx + 1}/{n_chunks} ({len(chunk_df)} imágenes)")
        
        chunk_descriptors = []
        
        for img in tqdm(image_generator(root, chunk_df), 
                       desc=f"Chunk {chunk_idx + 1}", 
                       total=len(chunk_df)):
            t0 = time()
            
            kp, desc = dense_sift_optimized(img, debug=debug)
            
            t1 = time()
            tiempos.append(t1 - t0)
            memorias.append(proceso.memory_info().rss / (1024 * 1024))
            
            if desc is not None and len(desc) > 0:
                # Limitar descriptores por imagen para balancear
                if len(desc) > 1000:
                    idx = np.random.choice(len(desc), 1000, replace=False)
                    desc = desc[idx]
                chunk_descriptors.append(desc)
        
        if chunk_descriptors:
            chunk_descriptors = np.vstack(chunk_descriptors)
            
            # Limitar tamaño del chunk
            if len(chunk_descriptors) > config.DESCRIPTORS_PER_CHUNK:
                idx = np.random.choice(len(chunk_descriptors), 
                                     config.DESCRIPTORS_PER_CHUNK, 
                                     replace=False)
                chunk_descriptors = chunk_descriptors[idx]
            
            all_descriptors.append(chunk_descriptors)
        
        # Liberar memoria
        gc.collect()
    
    if all_descriptors:
        all_descriptors = np.vstack(all_descriptors)
    else:
        all_descriptors = np.empty((0, 128), dtype=np.float32)
    
    # Guardar en cache
    if use_cache and len(all_descriptors) > 0:
        descriptor_cache.save_descriptors(all_descriptors, cache_key)
    
    # Guardar métricas
    df_medidas = pd.DataFrame({
        "iter": np.arange(len(tiempos)),
        "tiempo": tiempos,
        "memoria_MB": memorias
    })
    df_medidas.to_csv("medicion_dense_sift_optimized.csv", index=False)
    
    return all_descriptors, tiempos, memorias

def dense_sift_optimized(image, step=8, patch_size=16, debug=False):
    """
    Versión optimizada de dense_sift con menos operaciones costosas
    """
    if image is None:
        return [], None
    
    # Reescalado más rápido
    image_resized = cv2.resize(image, (config.MAX_IMAGE_SIZE, config.MAX_IMAGE_SIZE), 
                              interpolation=cv2.INTER_AREA)
    
    if len(image_resized.shape) == 3:
        gray = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_resized
    
    h, w = gray.shape
    
    # Crear keypoints en rejilla (vectorizado)
    ys, xs = np.mgrid[patch_size:h-patch_size:step, patch_size:w-patch_size:step]
    keypoints = [cv2.KeyPoint(float(x), float(y), float(patch_size)) 
                for x, y in zip(xs.ravel(), ys.ravel())]
    
    if not keypoints:
        return [], None
    
    # Filtrado por gradiente (optimizado)
    keypoints = filtrar_gradiente_optimized(gray, keypoints)
    
    if not keypoints:
        return [], None
    
    # Calcular entropías de una vez
    entropy_vals = np.array([kp_entropy_score_fast(gray, kp, patch_size) 
                           for kp in keypoints])
    
    # Filtrar por percentil 75 de entropía
    if len(entropy_vals) > 0:
        thr = np.percentile(entropy_vals, 75)
        keypoints = [kp for kp, s in zip(keypoints, entropy_vals) if s >= thr]
    
    if not keypoints:
        return [], None
    
    # Calcular SIFT una sola vez
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.compute(gray, keypoints)
    
    if descriptors is None or len(descriptors) == 0:
        return keypoints, None
    
    # Filtrar por norma
    desc_norm = np.linalg.norm(descriptors, axis=1)
    mask = desc_norm > 0.2
    descriptors = descriptors[mask]
    keypoints = [kp for i, kp in enumerate(keypoints) if mask[i]]
    
    if len(descriptors) == 0:
        return keypoints, None
    
    # RootSIFT optimizado
    descriptors = rootsift_optimized(descriptors)
    
    return keypoints, descriptors

def filtrar_gradiente_optimized(gray, kps, percentil=30):
    """
    Versión optimizada del filtrado por gradiente
    """
    if not kps:
        return []
    
    # Calcular gradiente una sola vez
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 1.0)
    gx = cv2.Sobel(gray_blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    
    threshold = np.percentile(mag, percentil)
    
    # Vectorizar el cálculo de valores medios
    h, w = gray.shape
    kept = []
    
    for kp in kps:
        yy = int(round(kp.pt[1]))
        xx = int(round(kp.pt[0]))
        y1, y2 = max(0, yy-1), min(h, yy+2)
        x1, x2 = max(0, xx-1), min(w, xx+2)
        
        if y1 < y2 and x1 < x2:
            local_mean = np.mean(mag[y1:y2, x1:x2])
            if local_mean >= threshold:
                kept.append(kp)
    
    return kept

def rootsift_optimized(descriptors):
    """
    RootSIFT vectorizado y optimizado
    """
    desc = descriptors.astype(np.float32)
    desc += 1e-12  # Evitar división por cero
    
    # Normalización L1
    desc /= desc.sum(axis=1, keepdims=True)
    
    # Raíz cuadrada y normalización L2
    desc = np.sqrt(desc)
    desc /= np.linalg.norm(desc, axis=1, keepdims=True)
    
    return desc

# ==================== PIPELINE ENTRENAMIENTO OPTIMIZADO ====================

def incremental_pca_fit(descriptors, n_components=64, batch_size=10000):
    """
    PCA incremental para grandes conjuntos de datos
    """
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    
    # Ajustar por chunks si hay muchos descriptores
    if len(descriptors) > batch_size:
        for i in range(0, len(descriptors), batch_size):
            batch = descriptors[i:i+batch_size]
            ipca.partial_fit(batch)
    else:
        ipca.fit(descriptors)
    
    return ipca

def train_kmeans_minibatch(descriptors, n_clusters=512, batch_size=10000, random_state=42):
    """
    Entrenar K-Means con MiniBatch para mejor escalabilidad
    """
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        max_iter=100,
        n_init=3,
        random_state=random_state,
        verbose=1
    )
    
    # Entrenar por chunks si es necesario
    if len(descriptors) > batch_size:
        for i in range(0, len(descriptors), batch_size):
            batch = descriptors[i:i+batch_size]
            kmeans.partial_fit(batch)
    else:
        kmeans.fit(descriptors)
    
    return kmeans

def features_para_kmeans_optimized(df, root, subset_size=2000, prop_test=0.2, 
                                 random_state=42, use_cache=True, debug=False):
    """
    Pipeline optimizado para preparar características
    """
    train_df, test_df = split_train_test(
        df, size=subset_size, prop_test=prop_test, random_state=random_state
    )
    
    print(f"🎯 Entrenamiento: {len(train_df)} imágenes")
    print(f"🧪 Test: {len(test_df)} imágenes")
    
    # Extraer descriptores
    descriptors_train, _, _ = extract_descriptors_optimized(
        train_df, root, debug=debug, use_cache=use_cache
    )
    
    descriptors_test, _, _ = extract_descriptors_optimized(
        test_df, root, debug=debug, use_cache=use_cache
    )
    
    print(f"📊 Descriptores entrenamiento: {len(descriptors_train)}")
    print(f"📊 Descriptores test: {len(descriptors_test)}")
    
    # Aplicar PCA incremental
    pca = incremental_pca_fit(descriptors_train, config.PCA_COMPONENTS)
    joblib.dump(pca, "pca_sift_optimized.pkl")
    
    # Transformar y normalizar
    descriptors_train_pca = pca.transform(descriptors_train)
    descriptors_train_pca = normalize(descriptors_train_pca, norm="l2")
    
    descriptors_test_pca = pca.transform(descriptors_test)
    descriptors_test_pca = normalize(descriptors_test_pca, norm="l2")
    
    return descriptors_train_pca, descriptors_test_pca, pca

def visualize_inertia_optimized(desc_train, desc_test, Ks, sample_size=100000):
    """
    Visualización optimizada de inercia con muestreo
    """
    # Muestrear si hay muchos datos
    if len(desc_train) > sample_size:
        idx_train = np.random.choice(len(desc_train), sample_size, replace=False)
        desc_train = desc_train[idx_train]
    
    if len(desc_test) > sample_size:
        idx_test = np.random.choice(len(desc_test), sample_size, replace=False)
        desc_test = desc_test[idx_test]
    
    score_train = []
    score_test = []
    
    for K in tqdm(Ks, desc="Entrenando K-Means"):
        kmeans = train_kmeans_minibatch(desc_train, n_clusters=K)
        
        # Calcular inercia en muestras más pequeñas
        train_sample = desc_train[:min(50000, len(desc_train))]
        test_sample = desc_test[:min(50000, len(desc_test))]
        
        score_train.append(kmeans.score(train_sample) / len(train_sample))
        score_test.append(kmeans.score(test_sample) / len(test_sample))
        
        # Guardar modelo
        joblib.dump(kmeans, f"kmeans_optimized_{K}.pkl")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(Ks, score_train, marker="o", label="Train", linewidth=2)
    plt.plot(Ks, score_test, marker="x", label="Test", linewidth=2)
    plt.xlabel("Número de clusters K")
    plt.ylabel("Inercia (promedio)")
    plt.title("Análisis del Codo - K-Means Optimizado")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("elbow_analysis_optimized.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return score_train, score_test

# ==================== FUNCIONES ORIGINALES (MANTENIDAS) ====================

def image_generator(root, df, file_col="FILE"):
    for arx in df[file_col]:
        path = os.path.join(root, arx)
        if os.path.exists(path):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            yield img

def split_train_test(df, size=None, prop_test=0.1, random_state=42):
    size = min(size, len(df)) if size is not None else len(df)
    subset_df = df.sample(n=size, random_state=random_state)
    test_size = int(size * prop_test)
    test_df = subset_df[:test_size]
    train_df = subset_df[test_size:]
    return train_df, test_df

def kp_entropy_score_fast(gray, kp, patch_size=12):
    x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
    h, w = gray.shape
    
    x1 = max(0, x - patch_size)
    x2 = min(w, x + patch_size)
    y1 = max(0, y - patch_size)
    y2 = min(h, y + patch_size)
    
    patch = gray[y1:y2, x1:x2]
    
    if patch.size == 0 or patch.shape[0] != 2 * patch_size or patch.shape[1] != 2 * patch_size:
        return 0.0
    
    patch_flat = patch.ravel().astype(np.int32)
    hist = np.bincount(patch_flat, minlength=256).astype(float)
    hist_sum = hist.sum()
    
    if hist_sum == 0:
        return 0.0
    
    hist /= hist_sum
    nz = hist[hist > 0]
    return float(-(nz * np.log2(nz)).sum())

# ==================== MAIN OPTIMIZADO ====================

if __name__ == "__main__":
    root = "toy_dataset"  # Cambiado de "../../toy_dataset" a "toy_dataset"
    df = cargar_labels()
    
    print("🚀 Iniciando pipeline optimizado...")
    print(f"📁 Dataset: {len(df)} imágenes")
    
    # CONFIGURACIÓN MODIFICABLE
    subset_size = 1000      # Número de imágenes a procesar
    n_clusters = 64         # Número de clusters
    debug = False           # True para modo debug (más lento)
    
    # Opción 1: Análisis de rendimiento con subset pequeño
    df_train = df.sample(n=subset_size, random_state=config.RANDOM_STATE)
    
    print("🔍 Extrayendo descriptores (optimizado)...")
    descs, tiempos, memorias = extract_descriptors_optimized(
        df_train, root, debug=debug, use_cache=True
    )
    
    print(f"✅ Descriptores extraídos: {len(descs)}")
    
    # Plot de métricas si hay datos
    if len(tiempos) > 0:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(tiempos, alpha=0.7)
        plt.title("Tiempo por iteración")
        plt.xlabel("Iteración")
        plt.ylabel("Tiempo (s)")
        
        plt.subplot(1, 3, 2)
        plt.plot(memorias, alpha=0.7, color='purple')
        plt.title("Uso de memoria")
        plt.xlabel("Iteración")
        plt.ylabel("Memoria (MB)")
        
        plt.subplot(1, 3, 3)
        plt.scatter(memorias, tiempos, alpha=0.5)
        plt.title("Memoria vs Tiempo")
        plt.xlabel("Memoria (MB)")
        plt.ylabel("Tiempo (s)")
        
        plt.tight_layout()
        plt.savefig("metricas_optimizadas.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    # Opción 2: Entrenamiento completo
    if len(descs) > 1000:
        print("🎯 Entrenando K-Means optimizado...")
        
        # PCA y normalización
        pca = incremental_pca_fit(descs, config.PCA_COMPONENTS)
        descs_pca = pca.transform(descs)
        descs_pca = normalize(descs_pca, norm="l2")
        
        # Entrenar K-Means
        kmeans = train_kmeans_minibatch(
            descs_pca, 
            n_clusters=n_clusters,
            random_state=config.RANDOM_STATE
        )
        
        joblib.dump(kmeans, "kmeans_optimized_final.pkl")
        joblib.dump(pca, "pca_optimized_final.pkl")
        
        print("✅ Modelos guardados:")
        print("   - kmeans_optimized_final.pkl")
        print("   - pca_optimized_final.pkl")
        
        # Análisis del codo (opcional)
        if input("¿Ejecutar análisis del codo? (s/n): ").lower() == 's':
            Ks = [64, 128, 256, 512]
            visualize_inertia_optimized(descs_pca, descs_pca, Ks)
    
    print("🎉 Pipeline optimizado completado!")
