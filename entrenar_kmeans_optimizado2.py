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

# ==================== CONFIGURACIÓN MÁS SENSIBLE ====================
class Config:
    MAX_IMAGE_SIZE = 512
    DENSE_SIFT_STEP = 8    # Más denso para más descriptores
    DENSE_SIFT_PATCH_SIZE = 16  # Patch más pequeño
    PCA_COMPONENTS = 64
    CHUNK_SIZE = 500       # Chunks más pequeños para mejor manejo
    DESCRIPTORS_PER_IMAGE = 500  # Más descriptores por imagen
    DESCRIPTORS_PER_CHUNK = 100000  # Más descriptores por chunk
    N_CLUSTERS = 128       # Menos clusters para empezar
    RANDOM_STATE = 42

config = Config()

# ==================== CACHÉ DE DESCRIPTORES ====================
class DescriptorCache:
    def __init__(self, cache_dir="descriptor_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_key(self, df_subset):
        content = "".join(sorted(df_subset["FILE"].astype(str))) + f"_{len(df_subset)}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def save_descriptors(self, descriptors, cache_key):
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.npy")
        np.save(cache_file, descriptors)
        return cache_file
    
    def load_descriptors(self, cache_key):
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.npy")
        if os.path.exists(cache_file):
            return np.load(cache_file)
        return None

descriptor_cache = DescriptorCache()

# ==================== FUNCIONES MÁS SENSIBLES ====================

def image_generator(root, df, file_col="FILE"):
    """Generador de imágenes con mejor diagnóstico"""
    for arx in df[file_col]:
        path = os.path.join(root, arx)
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                yield img
            else:
                print(f"❌ Error leyendo: {path}")
                yield None
        else:
            print(f"❌ No encontrado: {path}")
            yield None

def dense_sift_high_density(image, step=8, patch_size=16):
    """Dense SIFT más denso para obtener más descriptores"""
    if image is None:
        return [], None
    
    try:
        # Reescalado
        h, w = image.shape[:2]
        max_size = config.MAX_IMAGE_SIZE
        scale = min(max_size / h, max_size / w)
        new_w, new_h = int(w * scale), int(h * scale)
        image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Escala de grises
        if len(image_resized.shape) == 3:
            gray = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_resized
        
        h, w = gray.shape
        
        # Keypoints MUY densos
        keypoints = []
        for y in range(patch_size, h-patch_size, step):
            for x in range(patch_size, w-patch_size, step):
                keypoints.append(cv2.KeyPoint(float(x), float(y), float(patch_size)))
        
        print(f"   Keypoints generados: {len(keypoints)}")
        
        if not keypoints:
            return [], None
        
        # SIFT con más tolerancia
        sift = cv2.SIFT_create(contrastThreshold=0.01, edgeThreshold=10)  # Menos estricto
        keypoints, descriptors = sift.compute(gray, keypoints)
        
        if descriptors is None or len(descriptors) == 0:
            print("   ❌ No se pudieron extraer descriptores")
            return [], None
        
        print(f"   Descriptores extraídos: {len(descriptors)}")
        
        # RootSIFT
        descriptors = descriptors.astype(np.float32)
        eps = 1e-12
        descriptors /= (descriptors.sum(axis=1, keepdims=True) + eps)
        descriptors = np.sqrt(descriptors)
        descriptors /= (np.linalg.norm(descriptors, axis=1, keepdims=True) + eps)
        
        return keypoints, descriptors
        
    except Exception as e:
        print(f"❌ Error en dense_sift: {e}")
        return [], None

def extract_descriptors_high_yield(df, root, use_cache=False):  # FORZAR REGENERACIÓN
    """
    Extracción de descriptores forzando regeneración
    """
    # Forzar regeneración eliminando cache existente
    cache_key = descriptor_cache.get_cache_key(df)
    cache_file = os.path.join("descriptor_cache", f"{cache_key}.npy")
    if os.path.exists(cache_file):
        print("🗑️  Eliminando cache anterior...")
        os.remove(cache_file)
    
    all_descriptors = []
    total_images = len(df)
    processed_images = 0
    successful_images = 0
    
    print(f"🎯 Procesando {total_images} imágenes...")
    
    # Probar con las primeras 10 imágenes para diagnóstico
    print("\n🔍 DIAGNÓSTICO (primeras 10 imágenes):")
    test_df = df.head(10)
    
    for i, (img, filename) in enumerate(zip(image_generator(root, test_df), test_df['FILE'])):
        print(f"\n🖼️  Imagen {i+1}: {filename}")
        if img is not None:
            kp, desc = dense_sift_high_density(img)
            if desc is not None:
                print(f"   ✅ {len(desc)} descriptores")
            else:
                print("   ❌ 0 descriptores")
        else:
            print("   ❌ Imagen no cargada")
    
    # Procesar todo el dataset
    n_chunks = (total_images + config.CHUNK_SIZE - 1) // config.CHUNK_SIZE
    
    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * config.CHUNK_SIZE
        end_idx = min((chunk_idx + 1) * config.CHUNK_SIZE, total_images)
        chunk_df = df.iloc[start_idx:end_idx]
        
        print(f"\n📦 Chunk {chunk_idx + 1}/{n_chunks} ({len(chunk_df)} imágenes)")
        
        chunk_descriptors = []
        chunk_processed = 0
        chunk_successful = 0
        
        for img in tqdm(image_generator(root, chunk_df), 
                       desc=f"Chunk {chunk_idx + 1}", 
                       total=len(chunk_df)):
            
            kp, desc = dense_sift_high_density(img)
            processed_images += 1
            chunk_processed += 1
            
            if desc is not None and len(desc) > 0:
                # Usar todos los descriptores (sin limitar)
                chunk_descriptors.append(desc)
                successful_images += 1
                chunk_successful += 1
        
        print(f"   ✅ Éxito: {chunk_successful}/{chunk_processed} imágenes")
        
        if chunk_descriptors:
            chunk_descriptors = np.vstack(chunk_descriptors)
            print(f"   📊 Descriptores en chunk: {len(chunk_descriptors)}")
            
            all_descriptors.append(chunk_descriptors)
        
        gc.collect()
    
    # Combinar todos los descriptores
    if all_descriptors:
        all_descriptors = np.vstack(all_descriptors)
        print(f"\n📊 ESTADÍSTICAS FINALES:")
        print(f"   ✅ Imágenes procesadas: {processed_images}")
        print(f"   ✅ Imágenes exitosas: {successful_images} ({successful_images/processed_images*100:.1f}%)")
        print(f"   ✅ Descriptores totales: {len(all_descriptors)}")
        print(f"   📈 Descriptores por imagen: {len(all_descriptors)/successful_images:.1f}")
        
        # Guardar en cache
        descriptor_cache.save_descriptors(all_descriptors, cache_key)
        print("💾 Descriptores guardados en cache")
    else:
        all_descriptors = np.empty((0, 128), dtype=np.float32)
        print("❌ No se extrajeron descriptores")
    
    return all_descriptors

def incremental_pca_large(descriptors, n_components=64, batch_size=50000):
    """PCA incremental"""
    print("🎛️  Aplicando PCA...")
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    
    if len(descriptors) > batch_size:
        n_batches = (len(descriptors) + batch_size - 1) // batch_size
        for i in tqdm(range(n_batches), desc="PCA batches"):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(descriptors))
            batch = descriptors[start:end]
            ipca.partial_fit(batch)
    else:
        ipca.fit(descriptors)
    
    return ipca

def train_kmeans_flexible(descriptors, n_clusters=None):
    """K-Means que ajusta automáticamente el número de clusters"""
    if n_clusters is None:
        # Calcular número óptimo de clusters basado en los datos
        n_clusters = min(512, max(64, len(descriptors) // 100))
    
    print(f"🎯 Entrenando K-Means con {n_clusters} clusters...")
    
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=min(100000, len(descriptors)),
        max_iter=100,
        n_init=3,
        random_state=42,
        verbose=1
    )
    
    kmeans.fit(descriptors)
    return kmeans

# ==================== MAIN CON REGENERACIÓN FORZADA ====================

if __name__ == "__main__":
    root = "toy_dataset"
    df = cargar_labels()
    
    print("🚀 INICIANDO PROCESAMIENTO CON REGENERACIÓN")
    print("=" * 50)
    print(f"📁 Dataset completo: {len(df)} imágenes")
    
    # Verificar contenido del dataset primero
    print("\n🔍 VERIFICANDO DATASET...")
    sample_files = df['FILE'].head(5).tolist()
    for f in sample_files:
        path = os.path.join(root, f)
        exists = os.path.exists(path)
        print(f"   {f}: {'✅' if exists else '❌'}")

    # Extraer descriptores FORZANDO REGENERACIÓN
    print("\n🔍 EXTRAYENDO DESCRIPTORES (REGENERACIÓN FORZADA)...")
    start_time = time()
    
    descs = extract_descriptors_high_yield(df, root, use_cache=False)
    
    extraction_time = time() - start_time
    print(f"⏱️  Tiempo de extracción: {extraction_time/60:.1f} minutos")
    
    # Entrenamiento flexible
    min_descriptors = 1000  # Mínimo absoluto
    if len(descs) > min_descriptors:
        print(f"\n🎯 ENTRENANDO CON {len(descs)} DESCRIPTORES")
        
        # Ajustar clusters según cantidad de descriptores
        if len(descs) < 10000:
            n_clusters = 64
        elif len(descs) < 50000:
            n_clusters = 128
        else:
            n_clusters = 256
        
        # Aplicar PCA
        pca_start = time()
        pca = incremental_pca_large(descs, config.PCA_COMPONENTS)
        descs_pca = pca.transform(descs)
        descs_pca = normalize(descs_pca, norm="l2")
        pca_time = time() - pca_start
        
        # Entrenar K-Means
        kmeans_start = time()
        kmeans = train_kmeans_flexible(descs_pca, n_clusters=n_clusters)
        kmeans_time = time() - kmeans_start
        
        # Guardar modelos
        print("\n💾 GUARDANDO MODELOS...")
        joblib.dump(kmeans, f"kmeans_{len(descs)}_descriptors.pkl")
        joblib.dump(pca, f"pca_{len(descs)}_descriptors.pkl")
        
        print("✅ MODELOS GUARDADOS:")
        print(f"   - kmeans_{len(descs)}_descriptors.pkl ({n_clusters} clusters)")
        print(f"   - pca_{len(descs)}_descriptors.pkl")
        
        # Resumen final
        total_time = time() - start_time
        print(f"\n🎉 PROCESAMIENTO COMPLETADO!")
        print("=" * 50)
        print(f"📊 RESUMEN:")
        print(f"   - Imágenes procesadas: {len(df)}")
        print(f"   - Descriptores extraídos: {len(descs)}")
        print(f"   - Clusters: {n_clusters}")
        print(f"   - Tiempo total: {total_time/60:.1f} minutos")
        
    else:
        print(f"\n❌ INSUFICIENTES DESCRIPTORES")
        print(f"   Se necesitan al menos {min_descriptors} descriptores")
        print(f"   Se obtuvieron: {len(descs)} descriptores")
        
        # Diagnóstico detallado
        print("\n🔧 DIAGNÓSTICO AVANZADO:")
        print("1. Verificar que las imágenes en toy_dataset/ sean válidas")
        print("2. Probar con OpenCV directamente:")
        print("   python3 -c \"")
        print("   import cv2")
        print("   img = cv2.imread('toy_dataset/1.jpg')")
        print("   print(f'Forma: {img.shape if img is not None else \\\"None\\\"}')")
        print("   \"")

    print("🎉 PIPELINE FINALIZADO!")
