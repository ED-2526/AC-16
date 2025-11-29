import pandas as pd
from glob import glob
import os
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import shutil

#filtrar imagenes por tamaño
# contar archivos por estilo
# encontrar el estilo con menos archivos
# crear una fucnion que divida entre train y test con la misma proporcion de estilos y que sean disjuntos, y que le pase por parametro la cantidad de muestas por estilo para train y para test
# quiero saber el nombre del archivo de las imagenes más pequeñas

def get_archivos(root):
    for estilo_path in glob(os.path.join(root, '*')):
        estilo = os.path.basename(estilo_path)
        return glob(os.path.join(os.path.join(estilo_path, estilo),'*.jpg'))

def get_tamano_imagen(archivo):

    with Image.open(archivo) as img:
        return img.size[0] * img.size[1]

def historgrama_tamaños_imagenes(root, draw_intervalo=None):
    tamaños = []
    archivos = get_archivos(root)
    for archivo in archivos:
        tamaños.append(get_tamano_imagen(archivo))
    pd.Series(tamaños).hist(bins=500)
    if draw_intervalo:
        plt.axvline(x=draw_intervalo[0], color='r', linestyle='--')
        plt.axvline(x=draw_intervalo[1], color='r', linestyle='--')
    plt.show()

def contar_archivos_por_estilo(root):
    estilos = {}
    for estilo_path in glob(os.path.join(root, '*')):
        estilo = os.path.basename(estilo_path)
        archivos = glob(os.path.join(os.path.join(estilo_path, estilo),'*.jpg'))
        estilos[estilo] = len(archivos)
    return estilos

def estilo_minimo(estilos_count):
    return min(estilos_count, key=estilos_count.get)

root = "../dataset/"
estilos_count = contar_archivos_por_estilo(root)
print("Cantidad de archivos por estilo:")
print(estilos_count)
total = sum(estilos_count.values())
print(f"Total de archivos: {total}")
min_estilo = estilo_minimo(estilos_count)
print(f"Estilo con menos archivos: {min_estilo} con {estilos_count[min_estilo]} archivos")
