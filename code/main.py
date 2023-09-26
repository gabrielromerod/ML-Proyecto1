import os
import pandas as pd
import numpy as np
import carga_data, Kdtree, R_Logistica, Kdtree, svm, arbol
import matplotlib.pyplot as plt

def metrics_summary(y_true, y_pred, verbose=True):
    classes = np.unique(y_true)
    confusion = np.zeros((len(classes), len(classes)))
    for i, class_true in enumerate(classes):
        for j, class_pred in enumerate(classes):
            confusion[i, j] = np.sum((y_true == class_true) & (y_pred == class_pred))
    
    TP = np.diag(confusion)
    FP = np.sum(confusion, axis=0) - TP
    FN = np.sum(confusion, axis=1) - TP
    precision = np.where(TP + FP == 0, 0, TP / (TP + FP))
    recall = TP / (TP + FN)
    f1_score = np.where(precision + recall == 0, 0, 2 * (precision * recall) / (precision + recall))
    
    results_df = pd.DataFrame({
        'Clase': classes,
        'Precisión': precision,
        'Recall': recall,
        'F1-Score': f1_score
    })
    
    # Contador de clases con precisión positiva
    count_positive_precision = sum(precision > 0)
    
    # Calcula la precisión promedio
    avg_precision = np.mean(precision)
    
    if verbose:
        print(results_df)
    
    return count_positive_precision, avg_precision, f1_score
    

def svm_ex():
    directorio_especies = r"C:\Users\lords\Desktop\ML-Proyecto1\code\images"
    nueva_res = (128, 128)
    carga_data.crear_carpetas(directorio_especies)
    carga_data.organizar_imagenes(directorio_especies, nueva_res, directorio_especies)
    imagenes, etiquetas = carga_data.leer_imagenes(directorio_especies)  # Lee las imágenes y sus etiquetas
    cortes = 3  # Define la cantidad de cortes que se realizarán en cada imagen para vectorizarla
    imagenes_vectorizadas = carga_data.vectorizar_imagenes(imagenes, cortes)  # Vectoriza las imágenes
    data = carga_data.crear_dataframes(imagenes_vectorizadas, etiquetas)  # Crea los dataframes para entrenamiento y prueba
    train_data, val_data, test_data = carga_data.dividir_datos(data)  # Divide los datos en conjuntos de entrenamiento y prueba
    X_train = train_data.drop(columns=['Etiqueta']).to_numpy()
    y_train = train_data['Etiqueta'].to_numpy()
    X_val = val_data.drop(columns=['Etiqueta']).to_numpy()
    y_val = val_data['Etiqueta'].to_numpy()
    X_test = test_data.drop(columns=['Etiqueta']).to_numpy()
    y_test = test_data['Etiqueta'].to_numpy()
    weights_extended, biases_extended, train_losses, val_losses = svm.train_ovr_bootstrap(X_train, y_train, X_val, y_val, num_epochs=1000)
    svm.plot_loss(train_losses, val_losses)
    y_pred = svm.predict_ovr(X_test, weights_extended, biases_extended)
    svm.metrics_summary(y_test, y_pred)

def arbol_ex():
    directorio_especies = r"C:\Users\lords\Desktop\ML-Proyecto1\code\images"
    nueva_res = (128, 128)
    carga_data.crear_carpetas(directorio_especies)
    carga_data.organizar_imagenes(directorio_especies, nueva_res, directorio_especies)
    imagenes, etiquetas = carga_data.leer_imagenes(directorio_especies)  # Lee las imágenes y sus etiquetas
    cortes = 3  # Define la cantidad de cortes que se realizarán en cada imagen para vectorizarla
    imagenes_vectorizadas = carga_data.vectorizar_imagenes(imagenes, cortes)  # Vectoriza las imágenes
    data = carga_data.crear_dataframes(imagenes_vectorizadas, etiquetas)  # Crea los dataframes para entrenamiento y prueba
    train_data, val_data, test_data = carga_data.dividir_datos(data)  # Divide los datos en conjuntos de entrenamiento y prueba
    
    X_train = train_data.drop(columns=['Etiqueta']).to_numpy()
    y_train = train_data['Etiqueta'].to_numpy()
    X_test = test_data.drop(columns=['Etiqueta']).to_numpy()
    y_test = test_data['Etiqueta'].to_numpy()

    tree = arbol.DT()
    tree.train(X_train, y_train)

    y_pred = tree.predict(X_test)

    metrics_summary(y_test, y_pred)

def rlogistica_exc(seed=None):
    directorio_especies = r"C:\Users\lords\Desktop\ML-Proyecto1\code\images"
    nueva_res = (128, 128)
    
    np.random.seed(seed)  # Establece la semilla
    
    carga_data.crear_carpetas(directorio_especies)
    carga_data.organizar_imagenes(directorio_especies, nueva_res, directorio_especies)
    imagenes, etiquetas = carga_data.leer_imagenes(directorio_especies)
    
    cortes = 3
    imagenes_vectorizadas = carga_data.vectorizar_imagenes(imagenes, cortes)
    data = carga_data.crear_dataframes(imagenes_vectorizadas, etiquetas)
    
    train_data, val_data, test_data = carga_data.dividir_datos(data)
    
    model = R_Logistica.RegresionLogisticaMultinomial(train_data.drop(columns=['Etiqueta']).to_numpy(), train_data['Etiqueta'].to_numpy(), alpha=0.01, epochs=1000)
    val_losses = model.train(x_val=val_data.drop(columns=['Etiqueta']).to_numpy(), y_val=val_data['Etiqueta'].to_numpy())    
    y_pred = model.predict(test_data.drop(columns=['Etiqueta']).to_numpy())
    
    # Aquí, calculamos el F1-Score promedio y lo devolvemos para usarlo como métrica de referencia
    count_positive_precision, avg_precision, f1_scores = metrics_summary(test_data['Etiqueta'].to_numpy(), y_pred, verbose=True)

    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.show() 

def Kdtree_exc():
    directorio_especies = r"C:\Users\lords\Desktop\ML-Proyecto1\code\images"
    nueva_res = (128, 128)
    carga_data.crear_carpetas(directorio_especies)
    carga_data.organizar_imagenes(directorio_especies, nueva_res, directorio_especies)
    imagenes, etiquetas = carga_data.leer_imagenes(directorio_especies)  # Lee las imágenes y sus etiquetas
    cortes = 3  # Define la cantidad de cortes que se realizarán en cada imagen para vectorizarla
    imagenes_vectorizadas = carga_data.vectorizar_imagenes(imagenes, cortes) # Vectoriza las imágenes
    data = carga_data.crear_dataframes(imagenes_vectorizadas, etiquetas)  # Crea los dataframes para entrenamiento y prueba
    train_data, val_data, test_data = carga_data.dividir_datos(data)  # Divide los datos en conjuntos de entrenamiento y prueba

    tree = Kdtree.Kdtree(imagenes_vectorizadas, etiquetas)

    tree.searchknn(imagenes_vectorizadas[21], 10)
    
if __name__ == "__main__":
    rlogistica_exc()