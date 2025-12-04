import numpy as np
import cv2
import pickle
from pathlib import Path
from tamaño_keypoint import SIFTScaleEstimator
from herraientas import *
from gestor_descriptor import GestorDescriptor
from tqdm import tqdm

"""
1. convertim la imatge a gris i fa resize
2. busquem els kp i els guardem en una llista (kp=[kp1, kp2, ..., kpn])
3. fem una llista amb els descriptors de cada kp (descriptors=[desck1, desck2, ..., desckn]), i els guardem mitjaçant el gestor en la carpeta corresponent a la img: ara tindrem tantes carpetes com img, i dins tants descriptors com kp te la img.
"""

def reescalar(image, max_size=512):
    h, w = image.shape[:2]
    if h > w:
        scale_factor = max_size / float(h)
    else:
        scale_factor = max_size / float(w)
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return resized_image, scale_factor


def dense_kps(gray, step=8, estimador:SIFTScaleEstimator=SIFTScaleEstimator()):
    gray, _ = reescalar(gray)
    h, w = gray.shape
    keypoints = []
    estimador.build_pyramid(gray)
    for y in range(0, h, step):
        for x in range(0,w, step):
            patch_size = estimador.estimate(x, y)
            keypoints.append(cv2.KeyPoint(float(x), float(y), float(patch_size)))
    return keypoints


def procesar_imagen(imagen_path, sift: cv2.SIFT, scale_estimator: SIFTScaleEstimator, step=8):
    img = cargar_imagen(imagen_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kp = dense_kps(gray, estimador=scale_estimator, step=step)
    _, descriptores = sift.compute(gray, kp)
    
    if descriptores is None:
        return None
    return descriptores

def procesar_carpeta_imagenes(carpeta_raiz, carpeta_salida, sobreescribir=False, existe_out=True):
    sift = cv2.SIFT_create()
    scale_estimator = SIFTScaleEstimator()
    gestor = GestorDescriptor()
    gestor.inicializar(carpeta_salida, existe=existe_out)
    e = len(listar_estilos(carpeta_raiz))
    for estilo, archivos in tqdm(get_archivos(carpeta_raiz), desc="Leyendo descriptores por estilo", total=e):
        for archivo in tqdm(archivos, "Leyedo descriptores por imagen", total=len(archivos)):
            if not sobreescribir:
                if gestor.archivo_in(archivo, estilo):
                    continue
            descriptores = procesar_imagen(archivo, sift, scale_estimator)
            if descriptores is not None:
                gestor.guardar_descriptor(archivo, estilo, descriptores)

if __name__ == "__main__":
    carpeta_raiz = "../test"
    carpeta_salida = "../descriptores_test"
    resultados = procesar_carpeta_imagenes(carpeta_raiz, carpeta_salida)
