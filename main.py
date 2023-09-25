import os
from carga_data import create_species_folders, organizar_imagenes, leer_imagenes, vectorizar_imagenes, crear_dataframes, dividir_datos
from procesamiento_img import VecIMage

def main():
    directorio_especies = "images"
    nueva_res = (128, 128)  # Esta línea fue agregada
    
    create_species_folders(directorio_especies)
    
    organizar_imagenes(directorio_especies, nueva_res)  # Corrección aquí
    
    imagenes, etiquetas = leer_imagenes(directorio_especies)
    
    cortes = 3  
    imagenes_vectorizadas = vectorizar_imagenes(imagenes, cortes)
    
    data = crear_dataframes(imagenes_vectorizadas, etiquetas)
   
    train_data, test_data = dividir_datos(data)
    
if __name__ == "__main__":
    main()
