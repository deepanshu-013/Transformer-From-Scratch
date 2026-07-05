from tokenizer import Tokenizer
from embedding import Embedding
from positional_encoding import PositionalEncoding
from SelfAttention import SelfAttention
from feedforward import Linear, ReLU

corpus = [
    "I love dogs.",
    "Dogs bark.",
    "Cats meow."
]

token = Tokenizer()

embedding = Embedding(5000, 128)

positional_encoding = PositionalEncoding()

self_attention = SelfAttention()

linear_layer_1 = Linear(128, 512)
relu = ReLU()
linear_layer_2 = Linear(512, 128)

token.fit(corpus)
input_ids = token.brew('I LOVE DOGS', 10)["input_ids"]
input_weights = embedding.forward(input_ids)
pos_end = positional_encoding.forward(input_weights.shape[0], input_weights.shape[1])
pos_end += input_weights
attention_score = self_attention.forward(pos_end)
linear_output_1 = linear_layer_1.forward(attention_score)
relu_output = relu.forward(linear_output_1)
linear_output_2 = linear_layer_2.forward(relu_output)


print(linear_output_2)
