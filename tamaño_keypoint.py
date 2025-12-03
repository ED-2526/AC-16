import numpy as np
from herraientas import *
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