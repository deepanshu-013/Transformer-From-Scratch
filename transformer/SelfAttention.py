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

        Q = X @ self.WQ
        K = X @ self.WK
        V = X @ self.WV

        attention_weight = Q @ K.T

        attention_weight /= np.sqrt(self.d_model)

        attention_weight = self.softmax(attention_weight)

        output = attention_weight @ V
        return output







