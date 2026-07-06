from positional_encoding import PositionalEncoding
from SelfAttention import SelfAttention
from feedforward import FeedForward

class Encoder:
    def __init__(self, input_dim=128, hidden_dim=512):
        self.positional_encoding = PositionalEncoding()
        self.self_attention = SelfAttention()
        self.feedforward = FeedForward(input_dim, hidden_dim)

    def forward(self, input_weights):
        positional_encoding = self.positional_encoding.forward(input_weights.shape[0], input_weights.shape[1])
        X = positional_encoding + input_weights
        attention_output = self.self_attention.forward(X)
        X = X + attention_output  # Residual connection
        #Layer Normalization will be added here
        feedforward_output = self.feedforward.forward(X)
        X = X + feedforward_output  # Residual connection
        #Layer Normalization will be added here
        return X


































