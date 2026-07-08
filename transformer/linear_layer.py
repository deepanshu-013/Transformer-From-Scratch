import numpy as np

class Linear:
    def __init__(self, embedding_dim, vocab_size):
        self.weights = np.random.randn(embedding_dim, vocab_size) * 0.02
        self.bias = np.zeros((1, vocab_size))

    def forward(self, X):
        return np.dot(X, self.weights) + self.bias
    