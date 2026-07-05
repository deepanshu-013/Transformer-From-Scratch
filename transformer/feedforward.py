import numpy as np

np.random.seed(13)  # For reproducibility

class Linear:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.02 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

class ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        return self.output
