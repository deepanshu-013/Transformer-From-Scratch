import numpy as np

class Embedding:

    def __init__(self, vocab_size, embedding_dim):
        self.weights = np.random.randn(vocab_size, embedding_dim) * 0.02

    def forward(self, token_ids):
        self.token_ids = token_ids
        return self.weights[token_ids]

    def backward(self, d_output):
        self.d_weights = np.zeros_like(self.weights) # We are using zeros because in the forward pass, if your input sentence is 5 tokens long, you only access 5 rows out of the 10,000 rows(for example) in the weights matrix. The other 9,995 rows were completely unused.

        np.add.at(self.d_weights, self.token_ids, d_output)
        #using add.at to avoid overwriting the gradient if a word has appeared twice or more.
