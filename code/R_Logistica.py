import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
import random

class Rlogistica:
    def __init__(self, x, y=0, alpha=0):
        self.x = x
        self.y = y
        self.alpha = alpha

    def Hiperplano(self, w):
        return np.dot(self.x, w)

    def S(self, w):
        result = 1 / (1 + np.exp((-1) * self.Hiperplano(w)))
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

if __name__ == '__main__':

    data = os.listdir('images')

    result = []
    for i in data:
        result.append(np.linalg.norm(np.asarray(Image.open('images/' + i)).flatten()))

    
