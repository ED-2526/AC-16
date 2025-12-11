import numpy as np
import joblib
from herraientas import *
import os

#============== Normalización ==============

def rootsift(descriptors, eps=1e-7):
    descriptors = descriptors / (descriptors.sum(axis=1, keepdims=True) + eps)
    descriptors = np.sqrt(descriptors)
    return descriptors

def l1(x, eps=1e-7):
    x = np.asarray(x, dtype=np.float64)
    s = x.sum(axis=1, keepdims=True) + eps
    return x / s

def l2(x, eps=1e-7):
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=1, keepdims=True) + eps
    return x / n

def no_norm(x, eps=1e-7):
    return np.asarray(x, dtype=np.float64)


class GestorDescriptor:
    def __init__(self, tipo = ".pkl"):
        self._outdir = None
        self._estilos = []
        self._archivos = {}
        self._norm = no_norm
        self._tipo = tipo

    def _limpiar_gestor(self):
        self._outdir = None
        self._estilos = []
        self._archivos = {}

    def _get_dir(self, estilo, archivo=None):
        path = os.path.join(self._outdir, estilo)
        if archivo:
            path = os.path.join(path, archivo)
        return path

    def inicializar(self, out_dir, estilos=None, existe=False):
        self._limpiar_gestor()
        self._outdir = out_dir
        if existe:
            self.actualizar_gestor()
        else:
            self.crear_directorios(estilos)

    def actualizar_gestor(self):
        estilos = listar_estilos(self._outdir)
        self.crear_directorios(estilos)
        for estilo, archivos in get_archivos(self._outdir, terminacion=".pkl"):
            for archivo in archivos:
                archivo = os.path.basename(archivo)
                self._archivos[archivo] = estilo

    def crear_directorios(self, estilos=None):
        os.makedirs(self._outdir, exist_ok=True)
        if estilos:
            self._estilos = estilos
        if self._estilos:
            for estilo in self._estilos:
                os.makedirs(self._get_dir(estilo), exist_ok=True)

    def nuevo_estilo(self, estilo):
        self._estilos.append(estilo)
        os.makedirs(self._get_dir(estilo), exist_ok=True)

    def existe_estilo(self, estilo):
        return estilo in self._estilos

    def convertir_pkl(self, archivo):
        if archivo.endswith(".jpg"):
            return archivo[:-4] + ".pkl"
        return archivo
    
    def guardar_descriptor(self, archivo, estilo, descriptor):
        if not self.existe_estilo(estilo):
            self.nuevo_estilo(estilo)
        archivo = os.path.basename(archivo)
        archivo = self.convertir_pkl(archivo)
        joblib.dump(descriptor,self._get_dir(estilo, archivo))
        self._archivos[archivo] = estilo

    def guardar_descriptores(self, archivos, estilos, descriptores):
        for archivo, estilo, descriptor in zip(archivos, estilos, descriptores):
            self.guardar_descriptor(archivo, estilo, descriptor)

    def _get_estilo(self, archivo):
        return self._archivos[archivo]

    def cargar_descriptor(self, archivo, estilo= None):
        archivo = os.path.basename(archivo)
        archivo = self.convertir_pkl(archivo)
        if not estilo:
            estilo = self._get_estilo(archivo)
        path = self._get_dir(estilo, archivo)
        return self._norm(joblib.load(path))

    def cargar_descriptores(self, archivos, estilos=None):
        if estilos:
            for archivo, estilo in zip(archivos, estilos):
                yield self.cargar_descriptor(archivo, estilo)
        else:
            for archivo in archivos:
                yield self.cargar_descriptor(archivo)

    def cargar_estilo(self, estilo):
        archivos = listar_imagenes_estilo(self._outdir, estilo, terminacion=".pkl")
        for archivo in archivos:
            yield self.cargar_descriptor(archivo, estilo)

    def cargar_todos(self):
        archivos = listar_archivos(self._outdir, terminacion=".pkl")
        return self.cargar_descriptores(archivos)

    def archivo_in(self, archivo, estilo=None):
        archivo = self.convertir_pkl(os.path.basename(archivo))
        if estilo:
            return archivo in [self.convertir_pkl(os.path.basename(arc)) for arc in listar_imagenes_estilo(self._outdir, estilo, terminacion=".pkl")]
        return archivo in listar_archivos(self._outdir, terminacion=".pkl")


    def __str__(self):
        return f"{self._outdir}\n{self._estilos}\n{self._archivos}"

    def get_estilos(self):
        return self._estilos

    def __len__(self):
        return len(self._archivos)

    def set_norm(self, norm):
        normas = {"l1": l1, "l2": l2, "root sift": rootsift}
        if norm and norm.lower() in normas:
            self._norm = normas[norm]

    def len_estilo(self, estilo):
        return len(os.listdir(self._get_dir(estilo)))



if __name__ == "__main__":
    #root = "../train"
    #dataset = "../descriptores"
    g = GestorDescriptor()
    g.inicializar("../dataset", existe=True)
    print(len(g))
    #desc = g.cargar_descriptor("232333.jpg")
    #print(type(desc[0,0]))