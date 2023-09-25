import os

import carga_data, Kdtree, R_Logistica, Kdtree

def main():
    """
    Función principal del programa. Realiza la carga de imágenes, las organiza, las vectoriza, crea un Kdtree y busca los k vecinos más cercanos.
    """
    directorio_especies = r"C:\Users\lords\Desktop\ML-Proyecto1\code\images"
    nueva_res = (128, 128)  # Esta línea fue agregada para definir la nueva resolución de las imágenes

    carga_data.crear_carpetas(directorio_especies)
    carga_data.organizar_imagenes(directorio_especies, nueva_res, directorio_especies)
    
    imagenes, etiquetas = carga_data.leer_imagenes(directorio_especies)  # Lee las imágenes y sus etiquetas
    
    cortes = 3  # Define la cantidad de cortes que se realizarán en cada imagen para vectorizarla
    imagenes_vectorizadas = carga_data.vectorizar_imagenes(imagenes, cortes)  # Vectoriza las imágenes
    
    data = carga_data.crear_dataframes(imagenes_vectorizadas, etiquetas)  # Crea los dataframes para entrenamiento y prueba
    
    train_data, val_data, test_data = carga_data.dividir_datos(data)  # Divide los datos en conjuntos de entrenamiento y prueba


    #Creacion del Kdtree
    Knn = Kdtree.Kdtree(imagenes_vectorizadas, etiquetas)  # Crea el Kdtree con las imágenes y sus etiquetas correspondientes

    #buscando los knn mas similares
    Knn.searchknn(imagenes[20], 10)  # Busca los 10 vecinos más cercanos a la imagen en la posición 20 del conjunto de imágenes

    Rlogistica = R_Logistica.RegresionLogisticaMultinomial(train_data['vec_img'], train_data['Etiqueta'])
    Rlogistica.train()

    
    
if __name__ == "__main__":
    main()
