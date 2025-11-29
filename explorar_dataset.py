import pandas as pd
from glob import glob
import os

# contar archivos por estilo
# encontrar el estilo con menos archivos
# crear una fucnion que divida entre train y test con la misma proporcion de estilos y que sean disjuntos, y que le pase por parametro la cantidad de muestas por estilo para train y para test

def contar_archivos_por_estilo(root):
    estilos = {}
    for estilo_path in glob(os.path.join(root, '*')):
        estilo = os.path.basename(estilo_path)
        archivos = glob(os.path.join(os.path.join(estilo_path, estilo),'*.jpg'))
        estilos[estilo] = len(archivos)
    return estilos

root = "../dataset/"
estilos_count = contar_archivos_por_estilo(root)
print("Cantidad de archivos por estilo:")
print(estilos_count)