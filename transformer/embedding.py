import numpy as np
import random as rnd


class Embedding:

    def __init__(self, vocab_size, embedding_dim):
        rnd.seed(42)

        self.weights = np.random.randn(vocab_size, embedding_dim) * 0.02

    def forward(self, token_ids):
        return self.weights[token_ids]