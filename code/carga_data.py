import re
from shutil import move
import os
import random
import numpy as np
import pandas as pd
from skimage.io import imread
import pywt
from PIL import Image
import shutil

nueva_res = (128, 128) 

def identify_species(directory="images"):
    species_set = set()
    try:
        filenames = os.listdir(directory)
    except FileNotFoundError:
        print(f"El directorio {directory} no existe.")
        return []
    pattern = re.compile(r'00(\d)0+|0(\d{1,2})0+')
    for filename in filenames:
        match = pattern.search(filename)
        if match:
            species_number = match.group(1) if match.group(1) else match.group(2)
            species_set.add(species_number)
    return sorted(list(species_set))

def create_species_folders(species_list, base_directory="images"):
    for species in species_list:
        folder_path = os.path.join(base_directory, f"Especie_{species}")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

def organize_images_by_species(directory="images"):
    filenames = os.listdir(directory)
    pattern = re.compile(r'00(\d)0+|0(\d{1,2})0+')
    for filename in filenames:
        match = pattern.search(filename)
        if match:
            species_number = match.group(1) if match.group(1) else match.group(2)
            source = os.path.join(directory, filename)
            destination_folder = os.path.join(directory, f"Especie_{species_number}")
            if os.path.exists(destination_folder):
                destination = os.path.join(destination_folder, filename)
                move(source, destination)
            else:
                print(f"La carpeta de destino {destination_folder} no existe.")

class VecIMage:
    def __init__(self, image):
        self.image = image

    def process(self, cortes):
        LL = self.image
        for i in range(cortes):
            LL, (LH, HL, HH) = pywt.dwt2(LL, 'haar')
        return LL.flatten().tolist()

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

def dividir_datos(data, val_size=0.15, test_size=0.15):
    especies = data['Etiqueta'].unique()
    train_indices = []
    val_indices = []
    test_indices = []

    for especie in especies:
        indices = data.index[data['Etiqueta'] == especie].tolist()
        random.shuffle(indices)
        val_index = int(len(indices) * val_size)
        test_index = int(len(indices) * (val_size + test_size))
        val_indices.extend(indices[:val_index])
        test_indices.extend(indices[val_index:test_index])
        train_indices.extend(indices[test_index:])
    
    train_data = data.iloc[train_indices]
    val_data = data.iloc[val_indices]
    test_data = data.iloc[test_indices]

    return train_data, val_data, test_data

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

if __name__ == "__main__":
    species_list = identify_species()
    create_species_folders(species_list)
    organize_images_by_species()