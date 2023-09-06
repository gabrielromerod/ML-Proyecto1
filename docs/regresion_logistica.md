# Regresión logística - Clasificación de Mariposas 🦋 

Parte del proyecto de clasificación de mariposas consiste en la implementación de un modelo de regresión logística.

Nota: En esta parte dar una introducción a la regresión logística y explicar su aplicación en el proyecto de manera teorica para en la siguiente parte demostrar su implementación en Python.

## Implementación de la regresión logística

El uso de clases fue predominante en nuestra propia versión de la regresión logística. Para ello, se crearon tres clases: `Rlogistica`, `Training`y `Testing`.

### Clase Rlogistica

### Clase Training

### Clase `Testing`

La clase `Testing` es la responsable de evaluar el rendimiento del modelo de regresión logística en el dataset de mariposas. Se inicializa con tres argumentos: `x_test`, `y_test` y `w`, que representan las características de prueba, las etiquetas de prueba y los pesos del modelo, respectivamente. La clase utiliza una instancia de la clase `Rlogistica` para calcular las predicciones, `y_pred`, utilizando la función sigmoide, `S`.

```python
class Testing:
    def __init__(self, x_test, y_test, w):
        self.x_test = x_test
        self.y_test = y_test
        self.w = w
        self.logist = Rlogistica(x_test, y_test, 0)

    def Testing(self):
        n = len(self.y_test)
        y_pred = self.logist.S(self.w)
        y_pred = np.round(y_pred)
        correct = np.sum(y_pred == self.y_test)
        print(f"Número de datos correctos: {correct}")
        accuracy = (correct / n) * 100
        print(f"Porcentaje de aciertos: {accuracy}%")
        print(f"Porcentaje de fallas: {100 - accuracy}%")
```

El método `Testing` calcula el número de predicciones correctas comparando `y_pred` con `y_test` y, posteriormente, calcula la precisión del modelo. Finalmente, imprime el número de aciertos, el porcentaje de aciertos y el porcentaje de fallas.