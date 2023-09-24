import numpy as np

class RegresionLogisticaMultinomial:
    def __init__(self, x, y, alpha=0.01, epochs=1000):
        self.x = x
        self.y = y
        self.alpha = alpha
        self.epochs = epochs
        self.num_classes = len(np.unique(y))  # Número de clases
        self.num_features = x.shape[1]        # Número de características
        self.weights = np.zeros((self.num_features, self.num_classes))

    def hiperplano(self):
        return np.dot(self.x, self.weights)

    def softmax(self):
        logits = self.hiperplano()
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probabilities

    def loss_function(self):
        probabilities = self.softmax()
        n = self.x.shape[0]
        loss = -np.sum(np.log(probabilities[range(n), self.y])) / n
        return loss

    def gradient(self):
        probabilities = self.softmax()
        n = self.x.shape[0]
        gradient = np.dot(self.x.T, (probabilities - np.eye(self.num_classes)[self.y])) / n
        return gradient
    
    def change_parameters(self, gradient, w):
        return w - self.alpha * gradient
    
    def train(self):
        for epoch in range(self.epochs):
            gradient = self.gradient()
            self.weights = self.change_parameters(gradient, self.weights)

    def predict(self, x_test):
        probabilities = np.dot(x_test, self.weights)
        predicted_classes = np.argmax(probabilities, axis=1)
        return predicted_classes

# Ejemplo de uso
x_train = np.array([[1, 2, 1, 2, 1], 
                    [2, 3, 2, 3, 2],
                    [3, 4, 3, 3, 2],
                    [4, 5, 4, 5, 4],
                    [5, 6, 5, 6, 5],
                    [6, 7, 0, 6, 7],
                    [7, 8, 1, 7, 8],
                    [8, 9, 2, 8, 9],
                    [9, 10, 3, 9, 10]]
                    )

y_train = np.array([0, 
                    0, 
                    0, 
                    1,
                    1,
                    1,
                    2,
                    2,
                    2])  # Clases: 0, 1, 2

modelo = RegresionLogisticaMultinomial(x_train, y_train)
modelo.train()

x_test = np.array([[2, 3, 2, 3, 2],
                   [2, 2, 2, 2, 2]])
predicted_classes = modelo.predict(x_test)
print("Clases predichas:", predicted_classes)
