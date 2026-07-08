from linear_layer import Linear

class LMHead:
    def __init__(self, vocab_size, embedding_dim):
        self.linear = Linear(embedding_dim, vocab_size)

    def forward(self, X):
        logits = self.linear.forward(X)
        return logits