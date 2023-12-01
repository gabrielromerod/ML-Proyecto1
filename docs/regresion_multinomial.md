# Clasificación de Mariposas utilizando Regresión Logística Multinomial 🦋 
El presente proyecto se centra en la emocionante tarea de clasificar diversas especies de mariposas, un desafío que involucra la aplicación de técnicas avanzadas de aprendizaje automático. En particular, abordamos esta tarea utilizando un enfoque de regresión logística multinomial.

La regresión logística es un algoritmo de aprendizaje supervisado ampliamente conocido, comúnmente utilizado para resolver problemas de clasificación binaria. Sin embargo, en este contexto, nos enfrentamos a la tarea de clasificar más de dos clases diferentes de mariposas, lo que requiere la extensión de nuestro modelo para manejar "n" clases. Para abordar esta complejidad, hemos optado por la implementación de la regresión logística multinomial.

A lo largo de este informe, exploraremos en detalle cómo la regresión logística multinomial nos permite llevar a cabo esta tarea de clasificación de mariposas con éxito, considerando tanto su funcionamiento interno como su aplicación práctica en este emocionante proyecto de investigación.

## Funcionamiento de la Regresión Logística Multinomial:

### Hiperplano
En el contexto de la regresión logística multinomial y otros modelos de clasificación, el hiperplano es un concepto geométrico que se utiliza para separar las diferentes clases de datos en un espacio de características. En este espacio, cada dimensión corresponde a una característica o atributo de los datos.

El hiperplano es una superficie que tiene una dimensión menos que el espacio en el que se encuentra. Por ejemplo, en un espacio bidimensional (dos características), un hiperplano sería una línea recta que separa dos clases. En un espacio tridimensional (tres características), un hiperplano sería un plano bidimensional que divide el espacio en dos regiones.

El objetivo de la regresión logística multinomial es encontrar el hiperplano óptimo que mejor separe las clases de datos. Esto se logra ajustando los parámetros del modelo (los pesos) de manera que el hiperplano tenga la mejor capacidad de separación posible.

- Implementación:
```python
def hiperplano(self):
    return np.dot(self.x, self.weights)
```
### Función Softmax
La función softmax se utiliza para convertir un conjunto de valores (a menudo llamados logits o puntuaciones) en una distribución de probabilidades sobre varias clases diferentes. Es especialmente útil en problemas de clasificación multiclase, donde se deben asignar probabilidades a múltiples clases en lugar de solo dos, como en la clasificación binaria.

La función softmax se define matemáticamente como sigue:
Dada una matriz de logits Z de forma m×n, donde m es el número de ejemplos y n es el número de clases, la función softmax calcula las probabilidades P de que cada ejemplo pertenezca a cada una de las n clases de la siguiente manera:
![Softmax](https://imgs.search.brave.com/lMajfuwzabVxlYRlYGQ0Pb9eJIfJm7YC1zpSGVXTCRg/rs:fit:500:0:0/g:ce/aHR0cHM6Ly9jZG4u/c2FuaXR5LmlvL2lt/YWdlcy92cjhncnU5/NC9wcm9kdWN0aW9u/LzU4MmE2YzUxNzAx/YmI1ODRjMWNkZDY2/NjJjYzM3NmI5Y2Fk/YjcxNjAtMjA0OHgx/MTUyLnBuZw)

La función softmax transforma los logits en una distribución de probabilidades de tal manera que las clases con logits más altos tendrán mayores probabilidades asociadas. Esto significa que la clase con el logit más alto será la clase más probable según el modelo.

- Implementación:
```python
def softmax(self):
    logits = self.hiperplano()
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return probabilities
```

### Función de pérdida
En la regresión logística multinomial, la función de pérdida se utiliza para evaluar cuán buenas son las predicciones del modelo en problemas de clasificación multiclase.
Durante el entrenamiento de la regresión logística multinomial, el objetivo es minimizar la función de pérdida. Esto se logra ajustando los pesos del modelo de manera que las predicciones se acerquen lo más posible a los valores reales.
- Implementación:
```python
    def loss_function(self):
        probabilities = self.softmax()
        n = self.x.shape[0]
        loss = -np.sum(np.log(probabilities[range(n), self.y])) / n
        return loss
```

### Gradiente
La función de gradiente en la regresión logística multinomial se utiliza para calcular la dirección y la magnitud del cambio necesario en los pesos del modelo para minimizar la función de pérdida y, por lo tanto, mejorar la calidad de las predicciones del modelo.
El gradiente se calcula para cada peso en el modelo en relación con la función de pérdida total. El gradiente es un vector que apunta en la dirección de máximo aumento de la función de pérdida, y su magnitud determina la tasa de cambio.

$\nabla L = \frac{1}{n} X^T \cdot (\hat{Y} - Y)$

Donde:
- $\nabla$ es el vector gradiente de la función de pérdida.

- n es el número de ejemplos de entrenamiento.

- X es la matriz de características de entrenamiento.

- $\hat{Y}$ es la matriz de probabilidades predichas por el modelo (resultado de la función softmax).

- Y es la matriz de etiquetas reales.

- Implementación:
```python
def gradient(self):
    probabilities = self.softmax()
    n = self.x.shape[0]
    gradient = np.dot(self.x.T, (probabilities - np.eye(self.num_classes)[self.y])) / n
    return gradient
```
### Change parameters
Durante el proceso de entrenamiento, utilizamos la función change_parameters para actualizar los pesos del modelo. Esta función toma el gradiente calculado con respecto a la función de pérdida y los pesos actuales como entrada, y realiza una actualización de los pesos utilizando una tasa de aprendizaje (α). La tasa de aprendizaje controla la velocidad a la que los pesos del modelo convergen hacia los valores óptimos. El objetivo es minimizar la función de pérdida y mejorar la precisión de las predicciones a medida que avanzamos en el proceso de entrenamiento.
- Implementación:
```python
def change_parameters(self, gradient, w):
    return w - self.alpha * gradient
```
### Train
La función train es responsable de entrenar el modelo de regresión logística multinomial. Durante el entrenamiento, se ajustan los pesos del modelo de acuerdo con los datos de entrenamiento proporcionados. El proceso implica iterar a través de los datos varias veces (llamadas "épocas") y actualizar los pesos para minimizar la función de pérdida. El entrenamiento se lleva a cabo de la siguiente manera:

Inicialización: Al comienzo del entrenamiento, los pesos del modelo se inicializan típicamente en cero o con valores aleatorios pequeños.

Iteraciones: Durante cada iteración, se toma una muestra aleatoria de los datos de entrenamiento (esto se conoce como "mini-lote" o "mini-batch"). Luego, se calcula el gradiente de la función de pérdida con respecto a los pesos actuales del modelo utilizando esta muestra.

Actualización de pesos: Los pesos del modelo se actualizan utilizando el gradiente calculado y una tasa de aprendizaje (α). Esta actualización mueve los pesos en la dirección que reduce la función de pérdida.

Repetición: Este proceso se repite durante varias épocas o hasta que se alcance un criterio de convergencia, como una pérdida mínima o un número máximo de iteraciones.
- Implementación:
```python
def train(self):
    n = len(self.x)
    for _ in range(self.epochs):
        sample_idx = np.random.choice(n, n, replace=True)
        x_train, y_sample = self.x[sample_idx], self.y[sample_idx]
        gradient = self.gradient()
        self.weights = self.change_parameters(gradient, self.weights)
```
### Predict
La función predict toma las características de prueba como entrada, calcula las probabilidades de pertenencia a cada clase utilizando el hiperplano ponderado por los pesos del modelo y selecciona la clase con la probabilidad más alta como la predicción final para cada instancia de prueba. Esto permite que el modelo realice predicciones sobre nuevos datos de entrada.
- Implementación:
```python
def predict(self, x_test):
    probabilities = np.dot(x_test, self.weights)
    predicted_classes = np.argmax(probabilities, axis=1)
    return predicted_classes
```