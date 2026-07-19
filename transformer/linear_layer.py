import numpy as np

class Linear:
    def __init__(self, embedding_weights):
        self.weights = embedding_weights.T
        self.biases = np.zeros((1, self.weights.shape[1]))

    def forward(self, X):
        self.X = X
        return np.dot(self.X, self.weights) + self.biases

    def backward(self, d_inputs):
        self.d_weights = np.dot(self.X.T, d_inputs)
        self.d_biases = np.sum(d_inputs, axis=0, keepdims=True)
        self.d_X = np.dot(d_inputs, self.weights.T)
        return self.d_X
