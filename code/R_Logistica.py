import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
import random

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
        n = len(self.x)
        for _ in range(self.epochs):
            sample_idx = np.random.choice(n, n, replace=True)
            x_train, y_sample = self.x[sample_idx], self.y[sample_idx]
            gradient = self.gradient()
            self.weights = self.change_parameters(gradient, self.weights)

    def predict(self, x_test):
        probabilities = np.dot(x_test, self.weights)
        predicted_classes = np.argmax(probabilities, axis=1)
        return predicted_classes