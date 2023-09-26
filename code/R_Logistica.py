import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class RegresionLogisticaMultinomial:
    def __init__(self, x, y, alpha=0.01, epochs=1000):
        self.x = x
        self.y = y - np.min(y)
        self.alpha = alpha
        self.epochs = epochs
        self.num_classes = len(np.unique(y))
        self.num_features = x.shape[1]
        self.weights = np.zeros((self.num_features, self.num_classes))
        self.losses = []

    def hiperplano(self):
        return np.dot(self.x, self.weights)

    def softmax(self, logits):
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probabilities

    def loss_function(self, logits, y_true):
        probabilities = self.softmax(logits)
        n = len(y_true)
        epsilon = 1e-15
        loss = -np.sum(np.log(probabilities[range(n), y_true] + epsilon)) / n
        return loss

    def gradient(self, logits, y_true):
        probabilities = self.softmax(logits)
        n = len(y_true)
        gradient = np.dot(self.x.T, (probabilities - np.eye(self.num_classes)[y_true])) / n
        return gradient
    
    def change_parameters(self, gradient, w):
        return w - self.alpha * gradient
    
    def train(self, x_val=None, y_val=None):
        n = len(self.x)
        val_losses = []  # Agregar una lista para almacenar las pérdidas de validación

        for _ in range(self.epochs):
            sample_idx = np.random.choice(n, n, replace=True)
            x_sample, y_sample = self.x[sample_idx], self.y[sample_idx]
            logits = np.dot(x_sample, self.weights)
            self.losses.append(self.loss_function(logits, y_sample))
            gradient = self.gradient(logits, y_sample)
            self.weights = self.change_parameters(gradient, self.weights)

            # Si se proporcionan datos de validación, calcula la pérdida de validación
            if x_val is not None and y_val is not None:
                val_logits = np.dot(x_val, self.weights)
                val_loss = self.loss_function(val_logits, y_val)
                val_losses.append(val_loss)

        return val_losses  # Devuelve las pérdidas de validación

    def predict(self, x_test):
        logits = np.dot(x_test, self.weights)
        predicted_classes = np.argmax(logits, axis=1)
        return predicted_classes

    def plot_loss(self):
        plt.plot(self.losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss vs. Epochs')
        plt.show()

