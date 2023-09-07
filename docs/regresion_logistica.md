# Regresión logística - Clasificación de Mariposas 🦋 

Parte del proyecto de clasificación de mariposas consiste en la implementación de un modelo de regresión logística.

La regresión logística es un algoritmo de aprendizaje supervisado utilizado principalmente para problemas de clasificación binaria, aunque también se puede extender a problemas de clasificación multiclase.
# Funcionamiento de la Regresión Logística:

## hiperplano
$h(x_i) = w_0*x_0 + w_1 * x_1 + ... +w_n*x_n$

Donde: 
- $w_0$ es el bias
- $w_1$ es el peso de la primera característica
- $x_1$ es la primera característica
- $n$ es el número de características

El hiperplano es la frontera de decisión que separa las clases. Nos sirve para poder clasificar los datos.

## Hipótesis
$s(x_i) = \frac{1}{1+e^{-h(x_i)}}$

Donde:
- $s(x_i)$ es la hipótesis
- $h(x_i)$ es el hiperplano

Esta función se conoce como función sigmoide o logística. Esta función toma valores de -infinito a infinito y los transforma en valores entre 0 y 1.

## Loss Function
![Loss Function](/docs/assets/Loss_function_graphic.png)
$L = -\sum_{i=1}^{n}[y_i*log(s(x_i))+(1-y_i)*log(1-s(x_i))]$

Nos sirve para calcular que tan bien está aprendiendo nuestro modelo. Si el valor de y es 1 y el modelo está aprendiendo bien, el valor de la función de pérdida será bajo. Si el modelo no está aprendiendo bien, el valor de la función de pérdida será alto. De manera similar pero inversa funciona cuando el valor de y es 0.

## Derivadas
$\frac{\partial L}{\partial w_j} =\frac{1}{n} \sum_{i=1}^{n}(y_i - s(x_i))*(-x_{ij})$

El uso de las derivadas nos sirve para poder actualizar los pesos y el bias de nuestro modelo.

# Aplicaciones de la Regresión Logística:

La regresión logística se utiliza en una amplia variedad de aplicaciones, incluyendo:

**Diagnóstico Médico:** Por ejemplo, para predecir si un paciente tiene una enfermedad o no en función de sus síntomas y resultados de pruebas.

**Detección de Spam:** Para clasificar correos electrónicos como spam o no spam en función de su contenido y características.

**Crédito y Evaluación de Riesgos:** Para determinar si un solicitante de crédito es un riesgo crediticio o no, basándose en sus antecedentes financieros.

**Marketing:** Para predecir si un cliente comprará un producto o responderá a una campaña de marketing.

**Análisis de Sentimientos:** Para analizar opiniones de usuarios y clasificarlas como positivas o negativas.

# Limitaciones de la Regresión Logística:

**Linealidad:** La regresión logística asume una relación lineal entre las características y la probabilidad logarítmica. Puede tener dificultades para modelar relaciones más complejas entre las variables.

**Suposición de Independencia:** Supone que las observaciones son independientes entre sí, lo que puede no ser cierto en todos los casos, como en datos de series temporales.

**No es adecuada para Clasificación Multiclase:** Aunque se puede extender para clasificación multiclase (usando técnicas como la regresión logística multinomial o la regresión logística ordinal), existen algoritmos específicos para esta tarea que pueden ser más eficientes.

**Sensible a Datos Desbalanceados:** Si las clases están desequilibradas en términos de cantidad de datos, la regresión logística puede sesgarse hacia la clase dominante.

**No maneja bien datos faltantes:** La regresión logística asume que los datos están completos y puede no manejar adecuadamente los valores faltantes

## Implementación de la regresión logística

El uso de clases fue predominante en nuestra propia versión de la regresión logística. Para ello, se crearon tres clases: `Rlogistica`, `Training`y `Testing`.

### Clase Rlogistica
La clase `Rlogistica` es la encargada de encapsular el funcionamientode la regresión logistica. Se inicializa con tres argumentos: `x`, `y` y `alpha`, que representan las características, las etiquetas y la tasa de aprendizaje, respectivamente.
```python
class Rlogistica:
    def __init__(self, x, y=0, alpha=0):
        self.x = x
        self.y = y
        self.alpha = alpha

    def Hiperplano(self, w):
        return np.dot(self.x, w)

    def S(self, w):
        result = 1 / (1 + np.exp((-1) * self.Hiperplano(w)))  # write your code here
        return result

    def Loss_function(self, w):
        n = len(self.y)
        A = self.y * np.log(self.S(w))
        B = (1 - self.y) * np.log(1 - self.S(w))
        C = np.sum(A + B)
        loss = (-1 / n) * C
        return loss

    def Derivatives(self, w):
        n = len(self.y)
        s_x = self.S(w)
        error = self.y - s_x
        D = -np.dot(error, self.x) / n
        return D

    def change_parameters(self, derivatives, w):
        return w - self.alpha * derivatives
```
El método `Hiperplano` representa la función matemática del hiperplano para obtener las predicciones. El método `S` representa la función sigmoide. El método `Loss_function` representa la función de pérdida. El método `Derivatives` representa las derivadas de la función de pérdida. Finalmente, el método `change_parameters` actualiza los pesos del modelo.

### Clase Training
La clase `Training` es la responsable de entrenar el modelo de regresión logística en el dataset de mariposas. Se inicializa con tres argumentos: `x_train`, `y_train`, `alpha` y `epochs`, que representan las características de entrenamiento, las etiquetas de entrenamiento, la tasa de aprendizaje y la cantidad de veces que se entrena el modelo, respectivamente. La clase utiliza una instancia de la clase `Rlogistica` para calcular las predicciones, 

```python
class Training:
    def __init__(self, x_trian, y_train, epochs, alpha):
        self.Rlogistica = Rlogistica(x_trian, y_train, alpha)
        self.epochs = epochs

    def startT(self):
        loss = []
        w = np.ones(self.Rlogistica.x.shape[1])
        for i in range(self.epochs):
            L = self.Rlogistica.Loss_function(w)
            dw = self.Rlogistica.Derivatives(w)
            w = self.Rlogistica.change_parameters(dw, w)
            loss.append(L)

        return w, loss
```
El método `startT` entrena el modelo de regresión logística. Para ello, inicializa los pesos del modelo en 1 y, posteriormente, actualiza los pesos `epochs` veces. Finalmente, retorna los pesos y la lista de pérdidas.
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