import os
import pandas as pd
import numpy as np
import carga_data, Kdtree, R_Logistica, Kdtree, svm, arbol

def main():
    """
    Función principal del programa. Realiza la carga de imágenes, las organiza, las vectoriza, crea un Kdtree y busca los k vecinos más cercanos.
    """
    directorio_especies = r"C:\Users\lords\Desktop\ML-Proyecto1\code\images"
    nueva_res = (128, 128)  # Esta línea fue agregada para definir la nueva resolución de las imágenes

    carga_data.crear_carpetas(directorio_especies)
    carga_data.organizar_imagenes(directorio_especies, nueva_res, directorio_especies)
    
    imagenes, etiquetas = carga_data.leer_imagenes(directorio_especies)  # Lee las imágenes y sus etiquetas
    
    cortes = 60  # Define la cantidad de cortes que se realizarán en cada imagen para vectorizarla
    imagenes_vectorizadas = carga_data.vectorizar_imagenes(imagenes, cortes)  # Vectoriza las imágenes
    
    data = carga_data.crear_dataframes(imagenes_vectorizadas, etiquetas)  # Crea los dataframes para entrenamiento y prueba
    
    train_data, val_data, test_data = carga_data.dividir_datos(data)  # Divide los datos en conjuntos de entrenamiento y prueba

    #Creacion del Kdtree
    # Knn = Kdtree.Kdtree(imagenes_vectorizadas, etiquetas)  # Crea el Kdtree con las imágenes y sus etiquetas correspondientes
 
    # #buscando los knn mas similares
    # Knn.searchknn(imagenes[20], 10)  # Busca los 10 vecinos más cercanos a la imagen en la posición 20 del conjunto de imágenes

    # X_train = train_data.drop(columns=['Etiqueta']).to_numpy()
    # y_train = val_data['Etiqueta'].to_numpy()

    # Rlogistica = R_Logistica.RegresionLogisticaMultinomial(X_train, y_train)

    # Rlogistica.train()

    # # Predecimos el conjunto de prueba
    # X_test = test_data.drop(columns=['Etiqueta']).to_numpy()
    # y_test = test_data['Etiqueta'].to_numpy()

    # y_pred = Rlogistica.predict(X_test)

    # # Calculamos la precisión
    # accuracy = np.sum(y_pred == y_test)

    # print(f"La precisión es: {accuracy}, de un total de {len(y_test)} imágenes de prueba")

 # Datos para el árbol de decisión
    X_train = train_data.drop(columns=['Etiqueta']).to_numpy()
    y_train = train_data['Etiqueta'].to_numpy()

    # Entrenando el árbol de decisión
    tree = arbol.DT()
    tree.train(X_train, y_train)

    # Predecimos el conjunto de prueba
    X_test = test_data.drop(columns=['Etiqueta']).to_numpy()
    y_test = test_data['Etiqueta'].to_numpy()

    y_pred = tree.predict(X_test)

    # Calculamos la precisión
    accuracy = np.sum(y_pred == y_test)

    print(f"La precisión del árbol de decisión es: {accuracy / len(y_test) * 100:.2f}% de un total de {len(y_test)} imágenes de prueba")
    
    
    # np.random.seed(0)
    # feature_columns = [col for col in train_data.columns if col != 'Etiqueta']
    # weights_extended, biases_extended = svm.train_ovr(train_data[feature_columns].values, train_data['Etiqueta'].values, num_epochs=1000)

    # svm.test_ovr(test_data[feature_columns].values, weights_extended, biases_extended, test_data['Etiqueta'].values)

if __name__ == "__main__":
    main()
