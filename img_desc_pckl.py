import numpy as np
import cv2
import pickle
from pathlib import Path

class SIFTScaleEstimator:
    def __init__(self, num_octaves=4, num_scales=3, sigma0=1.6):
        self.num_octaves = num_octaves
        self.num_scales = num_scales
        self.sigma0 = sigma0
        self.k = 2 ** (1.0 / num_scales)

    def build_pyramid(self, gray):
        gray = gray.astype(np.float32) / 255.0
        sigma_init = np.sqrt(self.sigma0**2 - 0.5**2)
        base = cv2.GaussianBlur(gray, (0,0), sigma_init)

        self.gauss = []
        self.dog = []
        self.sigmas = []

        for o in range(self.num_octaves):
            gaussians = []
            sigmas = []

            for i in range(self.num_scales + 3):
                sigma = self.sigma0 * (self.k ** i)
                if i == 0:
                    g = base
                else:
                    g = cv2.GaussianBlur(base, (0,0), sigma)
                gaussians.append(g)
                sigmas.append(sigma)

            self.gauss.append(gaussians)
            self.sigmas.append(sigmas)

            dogs = [gaussians[i+1] - gaussians[i] for i in range(len(gaussians)-1)]
            self.dog.append(dogs)

            # siguiente octava = imagen reducida
            base = cv2.resize(
                gaussians[self.num_scales], 
                (gaussians[self.num_scales].shape[1] // 2,
                 gaussians[self.num_scales].shape[0] // 2),
                interpolation=cv2.INTER_NEAREST
            )

    def estimate(self, x, y):
        """Devuelve el tamaño SIFT estimado por DoG en el punto (x,y)."""
        best_val = -np.inf
        best_sigma = None
        best_octave = None

        for o in range(self.num_octaves):
            scale = 2 ** o
            xo = int(x / scale)
            yo = int(y / scale)

            dogs = self.dog[o]
            sigmas = self.sigmas[o]

            h, w = dogs[0].shape
            if not (0 <= xo < w and 0 <= yo < h):
                continue

            for i, d in enumerate(dogs):
                v = abs(d[yo, xo])
                if v > best_val:
                    best_val = v
                    best_sigma = sigmas[i+1]   # DoG[i] = G[i+1] - G[i]
                    best_octave = o

        if best_sigma is None:
            return 0.0

        # Tamaño como lo define OpenCV → diámetro = 2 * sigma_real
        return 2 * best_sigma * (2 ** best_octave)

def cargar_imagen(path):
    """Carga una imagen desde el path"""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {path}")
    return img

def descriptor_a_binario(descriptor):
    """Convierte un descriptor SIFT de 128 valores en binario (1s y 0s)"""
    media = np.mean(descriptor)
    descriptor_binario = (descriptor > media).astype(np.uint8)
    return descriptor_binario

def procesar_imagen(imagen_path, sift, scale_estimator):
    """Procesa una imagen y retorna la matriz de descriptores binarios"""
    
    # Cargar imagen
    img = cargar_imagen(imagen_path)
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Construir pirámide para estimación de escala
    scale_estimator.build_pyramid(gray)
    
    # Detectar keypoints SIFT
    kp = sift.detect(gray, None)
    
    # Estimar escalas mejoradas
    new_kp = []
    for p in kp:
        size_est = scale_estimator.estimate(int(p.pt[0]), int(p.pt[1]))
        new_kp.append(cv2.KeyPoint(p.pt[0], p.pt[1], size_est))
    
    # Calcular descriptores para los keypoints mejorados
    _, descriptores = sift.compute(gray, new_kp)
    
    if descriptores is None:
        return np.array([])  # Retornar matriz vacía si no hay descriptores
    
    # Convertir descriptores a binario
    matriz_descriptores = []
    for descriptor in descriptores:
        descriptor_binario = descriptor_a_binario(descriptor)
        matriz_descriptores.append(descriptor_binario)
    
    return np.array(matriz_descriptores)

def procesar_carpeta_imagenes(carpeta_raiz, archivo_salida="descriptores.pkl"):
    """
    Procesa todas las imágenes en una carpeta y subcarpetas
    """
    sift = cv2.SIFT_create()
    scale_estimator = SIFTScaleEstimator()
    
    resultados = {}
    
    # Buscar todas las imágenes en la carpeta y subcarpetas
    extensiones = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    for extension in extensiones:
        patron = f"**/*{extension}"
        paths_imagenes = list(Path(carpeta_raiz).glob(patron))
        
        for path_imagen in paths_imagenes:
            print(f"Procesando: {path_imagen}")
            
            try:
                # Procesar imagen
                matriz_descriptores = procesar_imagen(str(path_imagen), sift, scale_estimator)
                
                # Guardar resultado
                resultados[str(path_imagen)] = {
                    'matriz_descriptores': matriz_descriptores,
                    'num_keypoints': len(matriz_descriptores),
                    'shape': matriz_descriptores.shape if len(matriz_descriptores) > 0 else (0, 128)
                }
                
                print(f"  - Keypoints encontrados: {len(matriz_descriptores)}")
                
            except Exception as e:
                print(f"  - Error procesando {path_imagen}: {e}")
    
    # Guardar resultados en archivo pickle
    with open(archivo_salida, 'wb') as f:
        pickle.dump(resultados, f)
    
    print(f"\nProcesamiento completado. Resultados guardados en: {archivo_salida}")
    print(f"Total de imágenes procesadas: {len(resultados)}")
    
    return resultados


carpeta_raiz = "C:/Users/Mariona/Downloads/archive"
resultados = procesar_carpeta_imagenes(carpeta_raiz, "descriptores_sift_binarios.pkl")
