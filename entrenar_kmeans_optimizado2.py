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

# ==================== CONFIGURACIÓN PARA GRAN ESCALA ====================
class Config:
    MAX_IMAGE_SIZE = 512
    DENSE_SIFT_STEP = 24  # Más espaciado para mayor velocidad con muchas imágenes
    DENSE_SIFT_PATCH_SIZE = 32
    PCA_COMPONENTS = 64
    CHUNK_SIZE = 1000     # Procesar 1000 imágenes por chunk
    DESCRIPTORS_PER_IMAGE = 100  # Máximo descriptores por imagen
    DESCRIPTORS_PER_CHUNK = 50000  # Máximo descriptores por chunk
    N_CLUSTERS = 512      # Más clusters para dataset grande
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

# ==================== FUNCIONES OPTIMIZADAS PARA GRAN ESCALA ====================

def image_generator(root, df, file_col="FILE"):
    """Generador de imágenes con manejo de errores"""
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

def dense_sift_fast(image, step=24, patch_size=32):
    """Dense SIFT ultra rápido para gran escala"""
    if image is None:
        return [], None
    
    try:
        # Reescalado rápido
        h, w = image.shape[:2]
        max_size = config.MAX_IMAGE_SIZE
        scale = min(max_size / h, max_size / w)
        new_w, new_h = int(w * scale), int(h * scale)
        image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Escala de grises
        if len(image_resized.shape) == 3:
            gray = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_resized
        
        h, w = gray.shape
        
        # Keypoints en rejilla muy espaciada
        keypoints = []
        for y in range(patch_size, h-patch_size, step):
            for x in range(patch_size, w-patch_size, step):
                keypoints.append(cv2.KeyPoint(float(x), float(y), float(patch_size)))
        
        if not keypoints:
            return [], None
        
        # SIFT directo sin filtros complejos
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.compute(gray, keypoints)
        
        if descriptors is None or len(descriptors) == 0:
            return [], None
        
        # RootSIFT rápido
        descriptors = descriptors.astype(np.float32)
        eps = 1e-12
        descriptors /= (descriptors.sum(axis=1, keepdims=True) + eps)
        descriptors = np.sqrt(descriptors)
        descriptors /= (np.linalg.norm(descriptors, axis=1, keepdims=True) + eps)
        
        return keypoints, descriptors
        
    except Exception as e:
        print(f"❌ Error en dense_sift: {e}")
        return [], None

def extract_descriptors_large_scale(df, root, use_cache=True):
    """
    Extracción de descriptores optimizada para 26,000 imágenes
    """
    if use_cache:
        cache_key = descriptor_cache.get_cache_key(df)
        cached = descriptor_cache.load_descriptors(cache_key)
        if cached is not None:
            print("✅ Descriptores cargados desde cache")
            return cached
    
    all_descriptors = []
    total_images = len(df)
    processed_images = 0
    successful_images = 0
    
    print(f"🎯 Procesando {total_images} imágenes...")
    
    # Procesar por chunks para manejar memoria
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
            
            kp, desc = dense_sift_fast(img)
            processed_images += 1
            chunk_processed += 1
            
            if desc is not None and len(desc) > 0:
                # Limitar descriptores por imagen
                if len(desc) > config.DESCRIPTORS_PER_IMAGE:
                    idx = np.random.choice(len(desc), config.DESCRIPTORS_PER_IMAGE, replace=False)
                    desc = desc[idx]
                
                chunk_descriptors.append(desc)
                successful_images += 1
                chunk_successful += 1
        
        # Estadísticas del chunk
        print(f"   ✅ Éxito: {chunk_successful}/{chunk_processed} imágenes")
        
        if chunk_descriptors:
            chunk_descriptors = np.vstack(chunk_descriptors)
            
            # Limitar tamaño del chunk para evitar sobrecarga
            if len(chunk_descriptors) > config.DESCRIPTORS_PER_CHUNK:
                idx = np.random.choice(len(chunk_descriptors), 
                                     config.DESCRIPTORS_PER_CHUNK, 
                                     replace=False)
                chunk_descriptors = chunk_descriptors[idx]
                print(f"   📊 Chunk limitado a {len(chunk_descriptors)} descriptores")
            
            all_descriptors.append(chunk_descriptors)
        
        # Liberar memoria
        gc.collect()
    
    # Combinar todos los descriptores
    if all_descriptors:
        all_descriptors = np.vstack(all_descriptors)
        print(f"\n📊 ESTADÍSTICAS FINALES:")
        print(f"   ✅ Imágenes procesadas: {processed_images}")
        print(f"   ✅ Imágenes exitosas: {successful_images} ({successful_images/processed_images*100:.1f}%)")
        print(f"   ✅ Descriptores totales: {len(all_descriptors)}")
        print(f"   📈 Descriptores por imagen: {len(all_descriptors)/successful_images:.1f}")
    else:
        all_descriptors = np.empty((0, 128), dtype=np.float32)
        print("❌ No se extrajeron descriptores")
    
    # Guardar en cache
    if use_cache and len(all_descriptors) > 0:
        descriptor_cache.save_descriptors(all_descriptors, cache_key)
        print("💾 Descriptores guardados en cache")
    
    return all_descriptors

def incremental_pca_large(descriptors, n_components=64, batch_size=50000):
    """PCA incremental optimizado para grandes datasets"""
    print("🎛️  Aplicando PCA...")
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    
    # Ajustar por chunks grandes
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

def train_kmeans_large_scale(descriptors, n_clusters=512, batch_size=100000):
    """K-Means optimizado para gran escala"""
    print("🎯 Entrenando K-Means...")
    
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        max_iter=150,  # Más iteraciones para convergencia
        n_init=5,      # Más inicializaciones
        random_state=42,
        verbose=1
    )
    
    # Entrenar con barra de progreso
    if len(descriptors) > batch_size:
        n_batches = (len(descriptors) + batch_size - 1) // batch_size
        for i in tqdm(range(n_batches), desc="K-Means batches"):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(descriptors))
            batch = descriptors[start:end]
            kmeans.partial_fit(batch)
    else:
        kmeans.fit(descriptors)
    
    return kmeans

def visualize_cluster_statistics(kmeans, descriptors):
    """Visualizar estadísticas de los clusters"""
    print("📊 Analizando clusters...")
    
    # Predecir clusters para una muestra
    sample_size = min(100000, len(descriptors))
    sample_indices = np.random.choice(len(descriptors), sample_size, replace=False)
    sample = descriptors[sample_indices]
    labels = kmeans.predict(sample)
    
    # Calcular distribuciones
    unique, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.bar(unique[:20], counts[:20])  # Mostrar primeros 20 clusters
    plt.title("Top 20 Clusters (más poblados)")
    plt.xlabel("Cluster ID")
    plt.ylabel("Número de descriptores")
    plt.xticks(rotation=45)
    
    plt.subplot(1, 3, 2)
    plt.hist(counts, bins=50, alpha=0.7, color='green')
    plt.title("Distribución de tamaños de clusters")
    plt.xlabel("Tamaño del cluster")
    plt.ylabel("Frecuencia")
    
    plt.subplot(1, 3, 3)
    sorted_counts = np.sort(counts)[::-1]
    plt.plot(sorted_counts, 'o-', alpha=0.7)
    plt.title("Clusters ordenados por tamaño")
    plt.xlabel("Cluster (ordenado)")
    plt.ylabel("Tamaño")
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig("cluster_statistics.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📈 Estadísticas de clusters:")
    print(f"   - Clusters totales: {len(unique)}")
    print(f"   - Cluster más grande: {counts.max()} descriptores")
    print(f"   - Cluster más pequeño: {counts.min()} descriptores")
    print(f"   - Tamaño promedio: {counts.mean():.1f} descriptores")
    print(f"   - Desviación estándar: {counts.std():.1f}")

# ==================== MAIN PARA 26,000 IMÁGENES ====================

if __name__ == "__main__":
    root = "toy_dataset"
    df = cargar_labels()
    
    print("🚀 INICIANDO PROCESAMIENTO DE 26,000 IMÁGENES")
    print("=" * 50)
    print(f"📁 Dataset completo: {len(df)} imágenes")
    
    # Usar todas las imágenes
    df_full = df
    
    # Extraer descriptores de todas las imágenes
    print("\n🔍 EXTRAYENDO DESCRIPTORES...")
    start_time = time()
    
    descs = extract_descriptors_large_scale(df_full, root, use_cache=True)
    
    extraction_time = time() - start_time
    print(f"⏱️  Tiempo de extracción: {extraction_time/60:.1f} minutos")
    
    # Entrenamiento si tenemos suficientes descriptores
    if len(descs) > config.N_CLUSTERS * 10:  # Mínimo 10 descriptores por cluster
        print(f"\n🎯 ENTRENANDO CON {len(descs)} DESCRIPTORES")
        
        # Aplicar PCA
        pca_start = time()
        pca = incremental_pca_large(descs, config.PCA_COMPONENTS)
        descs_pca = pca.transform(descs)
        descs_pca = normalize(descs_pca, norm="l2")
        pca_time = time() - pca_start
        print(f"⏱️  Tiempo PCA: {pca_time:.1f} segundos")
        
        # Entrenar K-Means
        kmeans_start = time()
        kmeans = train_kmeans_large_scale(descs_pca, n_clusters=config.N_CLUSTERS)
        kmeans_time = time() - kmeans_start
        print(f"⏱️  Tiempo K-Means: {kmeans_time/60:.1f} minutos")
        
        # Guardar modelos
        print("\n💾 GUARDANDO MODELOS...")
        joblib.dump(kmeans, "kmeans_26000.pkl")
        joblib.dump(pca, "pca_26000.pkl")
        
        print("✅ MODELOS GUARDADOS:")
        print(f"   - kmeans_26000.pkl ({config.N_CLUSTERS} clusters)")
        print(f"   - pca_26000.pkl ({config.PCA_COMPONENTS} componentes)")
        
        # Visualizar resultados
        print("\n📈 GENERANDO ESTADÍSTICAS...")
        visualize_cluster_statistics(kmeans, descs_pca)
        
        # Resumen final
        total_time = time() - start_time
        print(f"\n🎉 PROCESAMIENTO COMPLETADO!")
        print("=" * 50)
        print(f"📊 RESUMEN:")
        print(f"   - Imágenes procesadas: {len(df_full)}")
        print(f"   - Descriptores extraídos: {len(descs)}")
        print(f"   - Clusters: {config.N_CLUSTERS}")
        print(f"   - Tiempo total: {total_time/60:.1f} minutos")
        print(f"   - Descriptores/segundo: {len(descs)/total_time:.1f}")
        
    else:
        print(f"\n❌ INSUFICIENTES DESCRIPTORES")
        print(f"   Se necesitan al menos {config.N_CLUSTERS * 10} descriptores")
        print(f"   Se obtuvieron: {len(descs)} descriptores")
        print("💡 Sugerencias:")
        print("   - Verificar que las imágenes sean válidas")
        print("   - Reducir el paso de Dense SIFT (step)")
        print("   - Aumentar el tamaño del patch")
    
    print("🎉 PIPELINE FINALIZADO!")
