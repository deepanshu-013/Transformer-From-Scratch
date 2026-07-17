import numpy as np

class SelfAttention:

    def __init__(self, d_model = 128):
        self.d_model = d_model
        self.WQ = np.random.randn(d_model, d_model) * 0.02
        self.WK = np.random.randn(d_model, d_model) * 0.02
        self.WV = np.random.randn(d_model, d_model) * 0.02

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x,axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def forward(self, X):
        self.X = X
        self.Q = X @ self.WQ
        self.K = X @ self.WK
        self.V = X @ self.WV

        raw_scores = self.Q @ self.K.T

        scaled_scores = raw_scores/np.sqrt(self.d_model)

        self.A = self.softmax(scaled_scores)

        self.output = self.A @ self.V
        return self.output

    def backward(self, doutput):
        # First operation
        d_A = doutput @ self.V.T
        d_V = self.A.T @ doutput

        # Softmax Backward
        sum_dA_A = np.sum(d_A * self.A, axis=1, keepdims=True)
        d_softmax_A = self.A * (d_A - sum_dA_A)

        # Scalar backward
        d_score = d_softmax_A / np.sqrt(self.d_model)

        # Score backward
        d_Q = d_score @ self.K
        d_K = d_score.T @ self.Q

        # Backward for WQ, WK, WV
        self.dWQ = self.X.T @ d_Q
        self.dWK = self.X.T @ d_K
        self.dWV = self.X.T @ d_V

        # Gradient for X
        dX_Q = d_Q @ np.transpose(self.WQ)
        dX_K = d_K @ np.transpose(self.WK)
        dX_V = d_V @ np.transpose(self.WV)

        # True gradient
        d_X = dX_Q + dX_K + dX_V
        return d_X










