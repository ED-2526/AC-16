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
from sklearn.cluster import KMeans
import hashlib
import pandas as pd
import gc

# ==================== CONFIGURACIÓN ====================
class Config:
    MAX_IMAGE_SIZE = 512
    DENSE_SIFT_STEP = 16  # Más espaciado para mayor velocidad
    DENSE_SIFT_PATCH_SIZE = 32
    PCA_COMPONENTS = 64
    CHUNK_SIZE = 100
    DESCRIPTORS_PER_CHUNK = 10000
    N_CLUSTERS = 64
    RANDOM_STATE = 42

config = Config()

# ==================== FUNCIONES OPTIMIZADAS ====================

def fast_entropy(gray_patch, bins=16):
    """Entropía rápida sin scikit-image"""
    if gray_patch.size == 0:
        return 0.0
    patch = gray_patch.ravel().astype(np.int32)
    hist = np.bincount(patch, minlength=256).astype(float)
    hist_sum = hist.sum()
    if hist_sum == 0:
        return 0.0
    hist /= (hist_sum + 1e-12)
    nz = hist[hist > 0]
    return float(-(nz * np.log2(nz)).sum())

def kp_entropy_score_fast(gray, kp, patch_size=12):
    """Calcula entropía para un keypoint"""
    x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
    h, w = gray.shape
    x1 = max(0, x - patch_size)
    x2 = min(w, x + patch_size)
    y1 = max(0, y - patch_size)
    y2 = min(h, y + patch_size)
    patch = gray[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    return fast_entropy(patch)

def image_generator(root, df, file_col="FILE"):
    """Generador de imágenes simple"""
    for arx in df[file_col]:
        path = os.path.join(root, arx)
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                yield img
        else:
            print(f"❌ No encontrado: {path}")

def dense_sift_simple(image, step=16, patch_size=32):
    """Dense SIFT simplificado y rápido"""
    if image is None:
        return [], None
    
    # Reescalado
    h, w = image.shape[:2]
    max_size = config.MAX_IMAGE_SIZE
    scale = min(max_size / h, max_size / w)
    new_w, new_h = int(w * scale), int(h * scale)
    image_resized = cv2.resize(image, (new_w, new_h))
    
    # Escala de grises
    if len(image_resized.shape) == 3:
        gray = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_resized
    
    h, w = gray.shape
    
    # Keypoints en rejilla
    keypoints = []
    for y in range(patch_size, h-patch_size, step):
        for x in range(patch_size, w-patch_size, step):
            keypoints.append(cv2.KeyPoint(float(x), float(y), float(patch_size)))
    
    if not keypoints:
        return [], None
    
    # SIFT directo
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.compute(gray, keypoints)
    
    if descriptors is None or len(descriptors) == 0:
        return keypoints, None
    
    # RootSIFT simple
    descriptors = descriptors.astype(np.float32)
    eps = 1e-12
    descriptors /= (descriptors.sum(axis=1, keepdims=True) + eps)
    descriptors = np.sqrt(descriptors)
    descriptors /= (np.linalg.norm(descriptors, axis=1, keepdims=True) + eps)
    
    return keypoints, descriptors

def extract_descriptors_simple(df, root, max_images=100):
    """Extracción de descriptores simplificada"""
    all_descriptors = []
    processed = 0
    
    print(f"🔍 Procesando {min(len(df), max_images)} imágenes...")
    
    for img in tqdm(image_generator(root, df.head(max_images)), 
                   desc="Extrayendo SIFT", 
                   total=min(len(df), max_images)):
        if img is not None:
            kp, desc = dense_sift_simple(img)
            if desc is not None and len(desc) > 0:
                # Limitar descriptores por imagen
                if len(desc) > 200:
                    idx = np.random.choice(len(desc), 200, replace=False)
                    desc = desc[idx]
                all_descriptors.append(desc)
                processed += 1
    
    if all_descriptors:
        all_descriptors = np.vstack(all_descriptors)
        print(f"✅ Procesadas {processed} imágenes, {len(all_descriptors)} descriptores")
    else:
        all_descriptors = np.empty((0, 128), dtype=np.float32)
        print("❌ No se extrajeron descriptores")
    
    return all_descriptors

def incremental_pca_fit(descriptors, n_components=64, batch_size=10000):
    """PCA incremental"""
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    if len(descriptors) > batch_size:
        for i in range(0, len(descriptors), batch_size):
            batch = descriptors[i:i+batch_size]
            ipca.partial_fit(batch)
    else:
        ipca.fit(descriptors)
    return ipca

def train_kmeans_minibatch(descriptors, n_clusters=64, batch_size=10000, random_state=42):
    """Entrenar K-Means con MiniBatch"""
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        max_iter=100,
        n_init=3,
        random_state=random_state,
        verbose=1
    )
    kmeans.fit(descriptors)
    return kmeans

# ==================== MAIN CORREGIDO ====================

if __name__ == "__main__":
    root = "toy_dataset"  # ← RUTA CORREGIDA
    df = cargar_labels()
    
    print("🚀 Iniciando pipeline K-Means optimizado...")
    print(f"📁 Dataset: {len(df)} imágenes")
    
    # CONFIGURACIÓN
    subset_size = 100       # Empezar con pocas imágenes para prueba
    n_clusters = 64
    
    # Procesar subset
    df_subset = df.head(subset_size)
    
    print("🔍 Extrayendo descriptores...")
    descs = extract_descriptors_simple(df_subset, root, max_images=subset_size)
    
    if len(descs) > n_clusters:
        print("🎯 Entrenando K-Means...")
        
        # Aplicar PCA
        pca = incremental_pca_fit(descs, config.PCA_COMPONENTS)
        descs_pca = pca.transform(descs)
        descs_pca = normalize(descs_pca, norm="l2")
        
        # Entrenar K-Means
        kmeans = train_kmeans_minibatch(
            descs_pca, 
            n_clusters=n_clusters,
            random_state=config.RANDOM_STATE
        )
        
        # Guardar modelos
        joblib.dump(kmeans, "kmeans_optimized.pkl")
        joblib.dump(pca, "pca_optimized.pkl")
        
        print("✅ Modelos guardados:")
        print("   - kmeans_optimized.pkl")
        print("   - pca_optimized.pkl")
        print(f"🎉 Entrenado con {len(descs)} descriptores de {subset_size} imágenes")
    else:
        print(f"❌ Muy pocos descriptores ({len(descs)}) para {n_clusters} clusters")
    
    print("🎉 Pipeline completado!")
