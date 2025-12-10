import numpy as np
import cv2

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

            # Siguiente octava (downsample)
            base = cv2.resize(
                gaussians[self.num_scales], 
                (gaussians[self.num_scales].shape[1] // 2,
                 gaussians[self.num_scales].shape[0] // 2),
                interpolation=cv2.INTER_NEAREST
            )

    def estimate(self, x, y):
        """
        Calcula el tamaño para múltiples puntos a la vez.
        x: array o lista de coordenadas X
        y: array o lista de coordenadas Y
        Retorna: array de numpy con los tamaños
        """
        # Asegurar que son arrays de numpy
        x = np.atleast_1d(np.array(x))
        y = np.atleast_1d(np.array(y))
        
        n_points = len(x)
        best_vals = np.full(n_points, -np.inf)
        final_sizes = np.zeros(n_points)
        
        for o in range(self.num_octaves):
            scale_factor = 2 ** o
            
            # Coordenadas proyectadas a la octava actual (vectorizado)
            xo = (x / scale_factor).astype(int)
            yo = (y / scale_factor).astype(int)
            
            # Verificar límites de imagen en esta octava
            if len(self.dog[o]) == 0: continue
            h, w = self.dog[o][0].shape
            
            # Máscara booleana de puntos válidos
            valid = (xo >= 0) & (xo < w) & (yo >= 0) & (yo < h)
            
            # Si ningún punto cae dentro en esta octava, saltamos
            if not np.any(valid):
                continue
            
            # Filtramos solo los índices válidos para no acceder fuera de matriz
            valid_xo = xo[valid]
            valid_yo = yo[valid]
            
            dogs = self.dog[o]
            sigmas = self.sigmas[o]
            
            for i, d in enumerate(dogs):
                # Extraemos valores de píxel masivamente
                vals = np.abs(d[valid_yo, valid_xo])
                
                # Comparamos con el mejor valor guardado hasta ahora
                # Nota: best_vals[valid] extrae los valores actuales para compararlos
                current_bests = best_vals[valid]
                improved = vals > current_bests
                
                # 'improved' es una máscara relativa a los puntos 'valid'.
                # Necesitamos actualizar 'best_vals' y 'final_sizes' en los índices originales.
                
                # Indices originales que son válidos y además han mejorado
                idx_to_update = np.where(valid)[0][improved]
                
                best_vals[idx_to_update] = vals[improved]
                
                # Cálculo del tamaño: 2 * sigma * 2^octava
                size_val = 2 * sigmas[i+1] * scale_factor
                final_sizes[idx_to_update] = size_val

        return final_sizes
    

if __name__ == "__main__":
    path = "C:/Users/Alex/OneDrive/Escritorio/paintings/train/Academic_Art/9223372032559872486.jpg"
    img = cargar_imagen(path)
    #visualizar_imagen(img)
    
    sift = cv2.SIFT_create()
    
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    scale_estimator = SIFTScaleEstimator()
    scale_estimator.build_pyramid(gray)
    
    # Detectar keypoints SIFT de OpenCV
    kp = sift.detect(gray, None)
    
    errors = []
    new_kp = []
    
    for p in kp:
        size_true = p.size
        size_est = scale_estimator.estimate(int(p.pt[0]), int(p.pt[1]))
        errors.append(abs(size_true - size_est))
        new_kp.append(cv2.KeyPoint(p.pt[0], p.pt[1], size_est))
    print(min(kp, key= lambda p: p.size).size)
    print(max(kp, key= lambda p: p.size).size)
    print(min(new_kp, key= lambda p: p.size).size)
    print(max(new_kp, key= lambda p: p.size).size)
    print("Error medio:", np.mean(errors))
    print("Error mediano:", np.median(errors))
    print("Número de keypoints comparados:", len(kp))
    
    img_cv2 = cv2.drawKeypoints(gray, kp, img.copy(),
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite('image_cv2.jpg', img_cv2)
    
    img_mia = cv2.drawKeypoints(gray, new_kp, img.copy(),
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite('image_mia2.jpg', img_mia)
    """
    # Applying SIFT detector
    sift = cv2.SIFT_create()
    kp = sift.detect(gray, None)
    
    # Marking the keypoint on the image using circles
    img=cv2.drawKeypoints(gray ,
                          kp ,
                          img ,
                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    cv2.imwrite('image-with-keypoints.jpg', img)
    """