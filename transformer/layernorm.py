import numpy as np 

class LayerNorm:
    def __init__(self, d_model, eps=1e-6):
        self.eps = eps
        self.gamma = np.ones((d_model,))
        self.beta = np.zeros((d_model,))

    def forward(self, X):
        self.X = X
        self.mean = np.mean(X, axis=1, keepdims=True)
        self.variance = np.var(X, axis=1, keepdims=True)
        self.centered = (X - self.mean)
        self.std = np.sqrt(self.variance + self.eps)
        self.normalized_X =  self.centered / self.std 
        return self.gamma * self.normalized_X + self.beta

    def backward(self, doutput):
        self.d_gamma = np.sum(doutput * self.normalized_X, axis=0)
        self.d_beta = np.sum(doutput, axis=0)
        d_normalized_X = doutput * self.gamma

        mean_d_normalized_X = np.mean(d_normalized_X, axis=1, keepdims=True)
        mean_d_normalized_X_normalized_X = np.mean(d_normalized_X * self.normalized_X, axis=1, keepdims=True)

        d_input = 1.0/self.std * (d_normalized_X - mean_d_normalized_X - self.normalized_X * mean_d_normalized_X_normalized_X) # Absolute Beauty!

        return d_input
