import numpy as np

class Linear:
    def __init__(self, embedding_dim, vocab_size):
        self.weights = np.random.randn(embedding_dim, vocab_size) * 0.02
        self.biases = np.zeros((1, vocab_size))

    def forward(self, X):
        self.X = X
        return np.dot(self.X, self.weights) + self.biases

    def backward(self, d_inputs):
        self.d_weights = np.dot(self.X.T, d_inputs)
        self.d_biases = np.sum(d_inputs, axis=0, keepdims=True)
        self.d_X = np.dot(d_inputs, self.weights.T)
        return self.d_X

    def update(self, learning_rate):
        self.weights -= learning_rate * self.d_weights
        self.biases -= learning_rate * self.d_biases

    def zero_grad(self):
        self.d_weights = np.zeros_like(self.weights)
        self.d_biases = np.zeros_like(self.biases)