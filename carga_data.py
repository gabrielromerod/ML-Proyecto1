import os
import random
import numpy as np
import pandas as pd
from skimage.io import imread
import pywt
from PIL import Image
import shutil
nueva_res = (128, 128) 

#1
class VecIMage:
    def __init__(self, image):
        self.image = image

    def process(self, cortes):
        LL = self.image
        for i in range(cortes):
            LL, (LH, HL, HH) = pywt.dwt2(LL, 'haar')
        return LL.flatten()

def esc_grises(path_img):
    imagen_color = Image.open(path_img)
    imagen_gris = imagen_color.convert("L") 
    imagen_array = np.array(imagen_gris)
    return imagen_array

def redimensionar(path_img, nueva_res):
    imagen = Image.open(path_img)
    imagen = imagen.resize(nueva_res)
    imagen.save(path_img)

def leer_imagenes(directorio_especies):
    imagenes = []
    etiquetas = []
    
    for especie_id in range(1, 11):
        especie_carpeta = os.path.join(directorio_especies, f"Especie_{especie_id}")
        for archivo in os.listdir(especie_carpeta):
            if archivo.endswith(".png"):
                ruta = os.path.join(especie_carpeta, archivo)
                imagen_en_gris = esc_grises(ruta)
                redimensionar(ruta, nueva_res) 
                imagenes.append(imagen_en_gris)
                etiquetas.append(especie_id)
    
    return imagenes, etiquetas

def vectorizar_imagenes(imagenes, cortes):
    imagenes_vectorizadas = []
    
    for imagen in imagenes:
        vec_image = VecIMage(imagen)
        vector = vec_image.process(cortes)
        imagenes_vectorizadas.append(vector)
    
    return imagenes_vectorizadas

def crear_dataframes(vec_img, etiquetas):
    data = pd.DataFrame(vec_img)
    data['Etiqueta'] = etiquetas
    return data

def dividir_datos(data, test_size=0.3):
    especies = data['Etiqueta'].unique()
    train_indices = []
    test_indices = []

    for especie in especies:
        indices = data.index[data['Etiqueta'] == especie].tolist()
        random.shuffle(indices)
        split_index = int(len(indices) * test_size)
        test_indices.extend(indices[:split_index])
        train_indices.extend(indices[split_index:])
    
    train_data = data.iloc[train_indices]
    test_data = data.iloc[test_indices]

    return train_data, test_data

def crear_carpetas(directorio_especies):
    for i in range(1, 11):
        especie_carpeta = os.path.join(directorio_especies, f"Especie_{i}")
        try:
            os.mkdir(especie_carpeta)
        except FileExistsError:
            pass

def organizar_imagenes(directorio_imagenes, nueva_res, directorio_especies):
    archivos = os.listdir(directorio_imagenes)

    for archivo in archivos:
        if archivo.endswith(".png"):
            num_especie = int(archivo[:3])

            new_carp = os.path.join(directorio_especies, f"Especie_{num_especie}")

            if not os.path.exists(new_carp):
                os.makedirs(new_carp)

            origen = os.path.join(directorio_imagenes, archivo)
            destino = os.path.join(new_carp, archivo)
            imagen = Image.open(origen)
            imagen = imagen.resize(nueva_res)
            imagen.save(destino)

            shutil.move(origen, destino)