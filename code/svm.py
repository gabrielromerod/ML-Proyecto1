import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def loss(y, x, w, bias, c):
    m = np.maximum(0, 1 - y * (x @ w + bias))
    return 0.5 * np.dot(w, w) + c * np.sum(m)

def grad(y, x, w, bias, c):
    margin = y * (np.dot(x, w) + bias)
    incorrect_classified = (margin < 1).astype(int)

    grad_w = w - c * np.dot(x.T, (incorrect_classified * y))
    grad_b = -c * np.sum(incorrect_classified * y)
 
    return grad_w, grad_b

def update(w, b, grad, alpha):
    w -= grad[0] * alpha
    b -= grad[1] * alpha
    return w, b

def train_ovr(x, y, num_epochs, c=1000, alpha=0.00001):
    n_features = x.shape[1]
    n_classes = len(np.unique(y))
    
    weights = np.zeros((n_classes, n_features))
    biases = np.zeros(n_classes)
    
    for current_class in range(n_classes):
        print(f"Training for class {current_class} vs rest...")
        # Creating a binary label for the current class
        binary_y = np.where(y == current_class, 1, -1)
        
        w = np.random.rand(n_features)
        b = np.random.random()
        
        for epoch in range(num_epochs):
            for idx, x_i in enumerate(x):
                grad_values = grad(binary_y[idx], x_i, w, b, c)
                w, b = update(w, b, grad_values, alpha)
        
        weights[current_class, :] = w
        biases[current_class] = b
        
    return weights, biases

def predict_ovr(x, weights, biases):
    scores = x @ weights.T + biases
    return np.argmax(scores, axis=1)

def test_ovr(x, weights, biases, y):
    predicted_labels = predict_ovr(x, weights, biases)
    print("Numero de aciertos:", np.sum(predicted_labels == y))
    print("Numero de errores:", np.sum(predicted_labels != y))
    print("Accuracy:", np.sum(predicted_labels == y) / len(y))

np.random.seed(544165)
iris_data_extended = pd.read_csv("Extended_DataSet_Iris_3_Clases.csv")

label_mapping_extended = {label: idx for idx, label in enumerate(iris_data_extended['variety'].unique())}
iris_data_extended['variety'] = iris_data_extended['variety'].map(label_mapping_extended)

msk_extended = np.random.rand(len(iris_data_extended)) < 0.8
train_data_extended = iris_data_extended[msk_extended]
test_data_extended = iris_data_extended[~msk_extended]

weights_extended, biases_extended = train_ovr(train_data_extended[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']].values, train_data_extended['variety'].values, num_epochs=1000)

test_ovr(test_data_extended[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']].values, weights_extended, biases_extended, test_data_extended['variety'].values)