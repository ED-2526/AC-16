import pandas as pd
from glob import glob
import os
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import shutil
import random
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
def split_dataset(
    root_dir,
    output_dir,
    train_ratio=0.8,
    seed=42
):
    random.seed(seed)
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    # Listar estilos (primer nivel)
    styles = [d for d in os.listdir(root_dir)
              if os.path.isdir(os.path.join(root_dir, d))]
    for style in styles:
        # carpeta donde realmente están las imágenes:
        inner_dir = os.path.join(root_dir, style, style)
        if not os.path.isdir(inner_dir):
            continue
        # Crear carpeta del estilo en train y test
        os.makedirs(os.path.join(train_dir, style), exist_ok=True)
        os.makedirs(os.path.join(test_dir, style), exist_ok=True)
        # Listar imágenes
        images = [f for f in os.listdir(inner_dir)
                  if os.path.isfile(os.path.join(inner_dir, f))]
        random.shuffle(images)
        N = len(images)
        N_train = int(train_ratio * N)
        train_imgs = images[:N_train]
        test_imgs = images[N_train:]
        # Copiar/mover train
        for img in train_imgs:
            src = os.path.join(inner_dir, img)
            dst = os.path.join(train_dir, style, img)
            shutil.move(src, dst)
        # Copiar/mover test
        for img in test_imgs:
            src = os.path.join(inner_dir, img)
            dst = os.path.join(test_dir, style, img)
            shutil.move(src, dst)

def fusionar_train_test(train_dir, test_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    styles = set()
    if os.path.isdir(train_dir):
        styles.update(os.listdir(train_dir))
    if os.path.isdir(test_dir):
        styles.update(os.listdir(test_dir))
    for style in styles:
        style_out = os.path.join(output_dir, style)
        os.makedirs(style_out, exist_ok=True)
        style_train = os.path.join(train_dir, style)
        style_test  = os.path.join(test_dir, style)
        def process_folder(folder):
            if os.path.isdir(folder):
                for f in os.listdir(folder):
                    src = os.path.join(folder, f)
                    if os.path.isfile(src):
                        dst = os.path.join(style_out, f)
                        shutil.move(src, dst)
        process_folder(style_train)
        process_folder(style_test)
root = "../dataset/"
estilos_count = contar_archivos_por_estilo(root)
print("Cantidad de archivos por estilo:")
print(estilos_count)
total = sum(estilos_count.values())
print(f"Total de archivos: {total}")
min_estilo = estilo_minimo(estilos_count)
print(f"Estilo con menos archivos: {min_estilo} con {estilos_count[min_estilo]} archivos")
split_dataset(
    root_dir=root,
    output_dir="../",
    train_ratio=0.8,
    seed=42
)