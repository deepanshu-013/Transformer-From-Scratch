import numpy as np

np.random.seed(13)  # For reproducibility

class FeedForward:
    def __init__(self, input_dim=128, hidden_dim=512):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.weights1 = 0.02 * np.random.randn(input_dim, hidden_dim)
        self.biases1 = np.zeros((1, hidden_dim))
        self.weights2 = 0.02 * np.random.randn(hidden_dim, input_dim)
        self.biases2 = np.zeros((1, input_dim))

    def forward(self, inputs):
        X = np.dot(inputs, self.weights1) + self.biases1
        X = np.maximum(0, X)  # ReLU activation
        X = np.dot(X, self.weights2) + self.biases2
        return X


