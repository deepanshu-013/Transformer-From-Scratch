import numpy as np

class Linear:
    def __init__(self, embedding_dim, vocab_size):
        self.weights = np.random.randn(embedding_dim, vocab_size) * 0.02
        self.biases = np.zeros((1, vocab_size))

    def forward(self, X):
        self.X = X
        return np.dot(self.X, self.weights) + self.biases

    def backward(self, d_logits):
        self.d_weights = np.dot(self.X.T, d_logits)
        self.d_biases = np.sum(d_logits, axis=0, keepdims=True)
        self.d_X = np.dot(d_logits, self.weights.T)