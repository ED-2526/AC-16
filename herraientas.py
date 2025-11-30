import cv2
import os
from PIL import Image
from matplotlib import pyplot as plt

def listar_estilos(root):
    return [
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ]
def listar_imagenes_estilo(root, estilo):
    archivos = []
    estilo_dir = os.path.join(root, estilo)
    for filename in os.listdir(estilo_dir):
        if filename.endswith(".jpg"):
            archivos.append(os.path.join(estilo_dir, filename))
    return archivos

def get_archivos(root):
    for estilo in listar_estilos(root):
        yield estilo, listar_imagenes_estilo(root, estilo)

def listar_archivos(root):
    archivos = []
    for estilo in listar_estilos(root):
        archivos.extend(listar_imagenes_estilo(root, estilo))
    return archivos

def get_tamano_imagen(archivo):
    with Image.open(archivo) as img:
        return img.size[0] * img.size[1], img.size[0] / img.size[1]

def cargar_imagen(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def visualizar_imagen(img):
    plt.imshow(img)
    plt.waitforbuttonpress()
    plt.close('all')