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

def train_ovr_bootstrap(x_train, y_train, x_val, y_val, num_epochs, c=1000, alpha=0.00001):
    n_features = x_train.shape[1]
    n_classes = len(np.unique(y_train))
    
    weights = np.zeros((n_classes, n_features))
    biases = np.zeros(n_classes)
    
    train_losses = []
    val_losses = []
    
    for current_class in range(n_classes):
        binary_y_train = np.where(y_train == current_class, 1, -1)
        binary_y_val = np.where(y_val == current_class, 1, -1)
        
        w = np.random.rand(n_features)
        b = np.random.random()
        
        for epoch in range(num_epochs):
            indices = np.random.choice(len(x_train), size=len(x_train), replace=True)
            x_bootstrap = x_train[indices]
            y_bootstrap = binary_y_train[indices]
            
            for idx, x_i in enumerate(x_bootstrap):
                grad_values = grad(y_bootstrap[idx], x_i, w, b, c)
                w, b = update(w, b, grad_values, alpha)
            
            train_loss = loss(binary_y_train, x_train, w, b, c)
            val_loss = loss(binary_y_val, x_val, w, b, c)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
        
        weights[current_class, :] = w
        biases[current_class] = b
        
    return weights, biases, train_losses, val_losses

def predict_ovr(x, weights, biases):
    scores = x @ weights.T + biases
    return np.argmax(scores, axis=1)

def plot_loss(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_losses, label='Entrenamiento', color='blue')
    plt.plot(epochs, val_losses, label='Validación', color='red')
    plt.title('Pérdida vs. Épocas')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid(True)
    plt.show()