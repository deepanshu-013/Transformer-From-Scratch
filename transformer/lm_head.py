from linear_layer import Linear

class LMHead:
    def __init__(self, embedding_weights):
        self.linear = Linear(embedding_weights)

    def forward(self, X):
        logits = self.linear.forward(X)
        return logits

    def backward(self, d_inputs):
        d_logits = self.linear.backward(d_inputs)
        return d_logits

    def update(self, learning_rate):
        self.linear.update(learning_rate)

    def zero_grad(self):
        self.linear.zero_grad()