import numpy as np 

class LayerNorm:
    def __init__(self, d_model, eps=1e-6):
        self.eps = eps
        self.gamma = np.ones((d_model,))
        self.beta = np.zeros((d_model,))

    def forward(self, X):
        mean = np.mean(X, axis=1, keepdims=True)
        variance = np.var(X, axis=1, keepdims=True)
        normalized_X = (X - mean) / np.sqrt(variance + self.eps)
        return self.gamma * normalized_X + self.beta