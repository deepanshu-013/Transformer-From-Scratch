from tokenizer import Tokenizer
from embedding import Embedding
from positional_encoding import PositionalEncoding

corpus = [
    "I love dogs.",
    "Dogs bark.",
    "Cats meow."
]

token = Tokenizer()

embedding = Embedding(5000, 128)

positional_encoding = PositionalEncoding()

token.fit(corpus)
input_ids = token.brew('I LOVE DOGS', 10)["input_ids"]
input_weights = embedding.forward(input_ids)
pos_end = positional_encoding.forward(input_weights.shape[0], input_weights.shape[1])


print(input_ids)
print(input_weights.shape)
print(pos_end.shape)

