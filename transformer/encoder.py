
from SelfAttention import SelfAttention
from feedforward import FeedForward
from layernorm import LayerNorm

class Encoder:
    def __init__(self, input_dim=128, hidden_dim=512):
        self.self_attention = SelfAttention(input_dim)
        self.feedforward = FeedForward(input_dim, hidden_dim)
        self.layer_norm_1 = LayerNorm(input_dim)
        self.layer_norm_2 = LayerNorm(input_dim)

    def forward(self, X, attention_mask):
        attention_output = self.self_attention.forward(X, attention_mask)
        X = self.layer_norm_1.forward(X + attention_output) # Layer Normalization 
        feedforward_output = self.feedforward.forward(X)
        X = self.layer_norm_2.forward(X + feedforward_output) # Layer Normalization
        return X

    def backward(self, d_output):
        d_layer_norm2 = self.layer_norm_2.backward(d_output)
        dX_ff =self.feedforward.backward(d_layer_norm2)
        d_residual2 = d_layer_norm2 + dX_ff
        d_layer_norm1 = self.layer_norm_1.backward(d_residual2)
        d_input= d_layer_norm1+ self.self_attention.backward(d_layer_norm1)

        return d_input

    def update(self, learning_rate):
        self.layer_norm_2.update(learning_rate)
        self.feedforward.update(learning_rate)
        self.layer_norm_1.update(learning_rate)
        self.self_attention.update(learning_rate)

    def zero_grad(self):
        self.self_attention.zero_grad()
        self.feedforward.zero_grad()
        self.layer_norm_1.zero_grad()
        self.layer_norm_2.zero_grad()
