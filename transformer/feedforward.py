import numpy as np

class FeedForward:
    def __init__(self, input_dim=128, hidden_dim=512):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.weights1 = 0.02 * np.random.randn(input_dim, hidden_dim)
        self.biases1 = np.zeros((1, hidden_dim))
        self.weights2 = 0.02 * np.random.randn(hidden_dim, input_dim)
        self.biases2 = np.zeros((1, input_dim))

    def forward(self, inputs):
        self.inputs = inputs
        self.layer1 = np.dot(self.inputs, self.weights1) + self.biases1
        self.activated_layer1 = np.maximum(0, self.layer1)  # ReLU activation
        X = np.dot(self.activated_layer1, self.weights2) + self.biases2
        return X

    def backward(self, d_values):
        # Layer 2 backward pass
        self.dweights2 = np.dot(self.activated_layer1.T, d_values)
        self.dbiases2 = np.sum(d_values, axis=0, keepdims=True)
        dactivated = np.dot(d_values, self.weights2.T)

         # RelU backward pass
        dactivated[self.layer1 <= 0] = 0

        # Layer 1 backward pass
        self.dweights1 = np.dot(self.inputs.T, dactivated)
        self.dbiases1 = np.sum(dactivated, axis=0, keepdims=True)
        doutput = np.dot(dactivated, self.weights1.T)

        return doutput
