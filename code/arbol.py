import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Nodo:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.feature = None
        self.threshold = None
        self.terminal = None

    def IsTerminal(self):
        return len(np.unique(self.Y)) == 1

    def best_split(self):
        best_gain = 0
        best_feature = None
        best_threshold = None
        current_entropy = self.Entropy(self.Y)

        for feature_index in range(self.X.shape[1]):
            thresholds = np.unique(self.X[:, feature_index])

            for threshold in thresholds:
                left_mask = self.X[:, feature_index] < threshold
                right_mask = ~left_mask

                left_entropy = self.Entropy(self.Y[left_mask])
                right_entropy = self.Entropy(self.Y[right_mask])

                weighted_avg_entropy = (left_mask.sum() * left_entropy + right_mask.sum() * right_entropy) / len(self.Y)
                info_gain = current_entropy - weighted_avg_entropy

                if info_gain > best_gain:
                    best_gain = info_gain
                    best_feature = feature_index
                    best_threshold = threshold

        self.feature = best_feature
        self.threshold = best_threshold
        return best_feature, best_threshold, best_gain

    def Entropy(self,Y):
        unique_labels, counts = np.unique(Y, return_counts=True)
        prob = counts / counts.sum()
        entropy_value = -np.sum(prob * np.log2(prob))
        return entropy_value

class DT:
    def __init__(self, bootstrap=False, max_depth=None):
        self.root = None
        self.bootstrap = bootstrap
        self.max_depth = max_depth

    def create_DT(self, node, depth=0):
        if node.IsTerminal() or (self.max_depth and depth == self.max_depth):
            node.terminal = np.bincount(node.Y).argmax()
            return

        best_feature, best_threshold, best_gain = node.best_split()

        if best_gain == 0:
            node.terminal = np.bincount(node.Y).argmax()
            return

        left_mask = node.X[:, best_feature] < best_threshold
        right_mask = ~left_mask

        left_node = Nodo(node.X[left_mask], node.Y[left_mask])
        right_node = Nodo(node.X[right_mask], node.Y[right_mask])

        node.left = left_node
        node.right = right_node

        self.create_DT(left_node, depth + 1)
        self.create_DT(right_node, depth + 1)

    def train(self, X, Y):
        if self.bootstrap:
            indices = np.random.choice(np.arange(X.shape[0]), size=X.shape[0], replace=True)
            X = X[indices]
            Y = Y[indices]

        self.root = Nodo(X, Y)
        self.create_DT(self.root)

    def predict_sample(self, node, sample):
        if node.terminal is not None:
            return node.terminal
        if sample[node.feature] < node.threshold:
            return self.predict_sample(node.left, sample)
        else:
            return self.predict_sample(node.right, sample)

    def predict(self, X):
        labels = [self.predict_sample(self.root, sample) for sample in X]
        return np.array(labels)
    
    def plot_tree(self, depths, train_errors, val_errors):
        plt.plot(depths, train_errors, label="Training Error")
        plt.plot(depths, val_errors, label="Validation Error")
        plt.xlabel("Tree Depth")
        plt.ylabel("Error")
        plt.legend()
        plt.title("Error vs. Tree Depth")
        plt.show()