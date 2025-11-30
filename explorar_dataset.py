import pandas as pd
from glob import glob
import os
from matplotlib import pyplot as plt
import numpy as np
import shutil
import random
import seaborn as sns
from herraientas import *

def aplanar_estilos(root_dir):
    for estilo in os.listdir(root_dir):
        estilo_dir = os.path.join(root_dir, estilo)
        if not os.path.isdir(estilo_dir):
            continue
        inner_dir = os.path.join(estilo_dir, estilo)
        if os.path.isdir(inner_dir):
            for f in os.listdir(inner_dir):
                src = os.path.join(inner_dir, f)
                dst = os.path.join(estilo_dir, f)
                if os.path.exists(dst):
                    continue
                shutil.move(src, dst)
            os.rmdir(inner_dir)


def historgrama_tamaños_imagenes(root):
    tamaños = []
    ratios = []
    for archivo in listar_archivos(root):
        tamaño, ratio = get_tamano_imagen(archivo)
        tamaños.append(tamaño)
        ratios.append(ratio)
    serie = pd.Series(tamaños)
    serie.hist(bins=1000)
    plt.show()
    print(serie.describe())
    serie = pd.Series(ratios)
    serie.hist(bins=1000)
    plt.show()
    print(serie.describe())

def contar_archivos_por_estilo(root):
    estilos = {}
    for estilo, archivos in get_archivos(root):
        estilos[estilo] = len(archivos)
    return estilos

def estilo_minimo(estilos_count):
    return min(estilos_count, key=estilos_count.get)

def split_dataset(
    root_dir,
    output_dir,
    train_ratio=0.8,
    seed=42,
    porcentaje_uso=0.45
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
        inner_dir = os.path.join(root_dir, style)
        # Crear carpeta del estilo en train y test
        os.makedirs(os.path.join(train_dir, style), exist_ok=True)
        os.makedirs(os.path.join(test_dir, style), exist_ok=True)
        images = [f for f in os.listdir(inner_dir)
                  if os.path.isfile(os.path.join(inner_dir, f))]
        random.shuffle(images)
        images = images[:int(len(images)*porcentaje_uso)]
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

if __name__ == "__main__":
    root = "../dataset"
    aplanar_estilos(root)
    split_dataset(root, "../")
    """
    estilos_count = contar_archivos_por_estilo(root)
    print("Cantidad de archivos por estilo:")
    print(estilos_count)
    total = sum(estilos_count.values())
    print({estilo: f"{100*valor/total:.2f}" for estilo, valor in estilos_count.items()})
    print(f"Total de archivos: {total}")
    min_estilo = estilo_minimo(estilos_count)
    print(f"Estilo con menos archivos: {min_estilo} con {estilos_count[min_estilo]} archivos")
    labels = estilos_count.keys()
    data = estilos_count.values()
    colors = sns.color_palette('bright')

    # plotting data on chart
    plt.pie(data, labels=labels, colors=colors, autopct='%.0f%%')
    plt.show()
    """