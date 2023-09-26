import numpy as np
import matplotlib.pyplot as plt

class RegresionLogisticaMultinomial:
    def __init__(self, x, y, alpha=0.01, epochs=1000, lambda_reg=0.1):
        self.x = np.insert(x, 0, 1, axis=1)  # Añadimos el término de sesgo
        self.y = y - np.min(y)
        self.alpha = alpha
        self.epochs = epochs
        self.lambda_reg = lambda_reg
        self.num_classes = len(np.unique(y))
        self.num_features = self.x.shape[1]
        self.weights = np.random.randn(self.num_features, self.num_classes) * 0.01  # Inicialización aleatoria
        self.losses = []

    def softmax(self, logits):
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def loss_function(self, logits, y_true):
        probabilities = self.softmax(logits)
        n = len(y_true)
        log_prob = np.log(probabilities[np.arange(n), y_true])
        loss = -np.mean(log_prob)
        
        l2_reg = (self.lambda_reg / 2) * np.sum(self.weights ** 2)
        loss += l2_reg
        
        return loss

    def gradient(self, x, logits, y_true):
        probabilities = self.softmax(logits)
        n = len(y_true)
        y_encoded = np.eye(self.num_classes)[y_true]
        
        gradient = np.dot(x.T, (probabilities - y_encoded)) / n
        
        gradient += self.lambda_reg * self.weights
        
        return gradient

    def train(self, x_val=None, y_val=None, bootstrap_size=None):
        val_losses = []
        n_samples = self.x.shape[0]
        
        if bootstrap_size is None:
            bootstrap_size = n_samples

        for epoch in range(self.epochs): 
            bootstrap_indices = np.random.choice(np.arange(n_samples), size=bootstrap_size, replace=True)
            x_bootstrap = self.x[bootstrap_indices]
            y_bootstrap = self.y[bootstrap_indices]
            
            logits = np.dot(x_bootstrap, self.weights)
            self.losses.append(self.loss_function(logits, y_bootstrap))
            
            gradient = self.gradient(x_bootstrap, logits, y_bootstrap)
            self.weights -= self.alpha * gradient

            if x_val is not None and y_val is not None:
                x_val_bias = np.insert(x_val, 0, 1, axis=1)
                val_logits = np.dot(x_val_bias, self.weights)
                val_loss = self.loss_function(val_logits, y_val - np.min(y_val))
                val_losses.append(val_loss)

        return val_losses

    def predict(self, x_test):
        x_test_bias = np.insert(x_test, 0, 1, axis=1)
        logits = np.dot(x_test_bias, self.weights)
        return np.argmax(logits, axis=1)

    def plot_loss(self, val_losses=None):
        plt.plot(self.losses, label="Entrenamiento")
        if val_losses is not None:
            plt.plot(val_losses, label="Validación")
        plt.title("Función de pérdida")
        plt.xlabel("Épocas")
        plt.ylabel("Pérdida")
        plt.legend()
        plt.show()