import os
import random
import numpy as np
import pandas as pd
from skimage.io import imread
import pywt
from PIL import Image
import shutil
nueva_resolucion = (128, 128) 


class VecIMage:
    def __init__(self, image):
        self.image = image

    def process(self, cortes):
        LL = self.image
        for i in range(cortes):
            LL, (LH, HL, HH) = pywt.dwt2(LL, 'haar')
        return LL.flatten()

def convertir_a_escala_de_grises(ruta_imagen):
    imagen_color = Image.open(ruta_imagen)
    imagen_gris = imagen_color.convert("L") 
    imagen_array = np.array(imagen_gris)
    return imagen_array

def redimensionar_imagen(ruta_imagen, nueva_resolucion):
    imagen = Image.open(ruta_imagen)
    imagen = imagen.resize(nueva_resolucion)
    imagen.save(ruta_imagen)

def leer_imagenes(directorio_especies):
    imagenes = []
    etiquetas = []
    
    for especie_id in range(1, 11):
        especie_carpeta = os.path.join(directorio_especies, f"Especie_{especie_id}")
        for archivo in os.listdir(especie_carpeta):
            if archivo.endswith(".png"):
                ruta = os.path.join(especie_carpeta, archivo)
                imagen_en_gris = convertir_a_escala_de_grises(ruta)
                redimensionar_imagen(ruta, nueva_resolucion) 
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

def crear_dataframes(imagenes_vectorizadas, etiquetas):
    data = pd.DataFrame(imagenes_vectorizadas)
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

def organizar_imagenes(directorio_imagenes, nueva_resolucion, directorio_especies):
    archivos = os.listdir(directorio_imagenes)

    for archivo in archivos:
        if archivo.endswith(".png"):
            numero_especie = int(archivo[:3])

            carpeta_destino = os.path.join(directorio_especies, f"Especie_{numero_especie}")

            if not os.path.exists(carpeta_destino):
                os.makedirs(carpeta_destino)

            origen = os.path.join(directorio_imagenes, archivo)
            destino = os.path.join(carpeta_destino, archivo)
            imagen = Image.open(origen)
            imagen = imagen.resize(nueva_resolucion)
            imagen.save(destino)

            shutil.move(origen, destino)

def main():
    directorio_especies = "images"
    
    
    crear_carpetas(directorio_especies)
    
    organizar_imagenes(directorio_especies, nueva_resolucion, directorio_especies)
    
    imagenes, etiquetas = leer_imagenes(directorio_especies)
    
    cortes = 3  
    imagenes_vectorizadas = vectorizar_imagenes(imagenes, cortes)
    
    data = crear_dataframes(imagenes_vectorizadas, etiquetas)
   
    train_data, test_data = dividir_datos(data)
    
if __name__ == "__main__":
    main()
