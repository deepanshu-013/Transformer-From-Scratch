from tokenizer import Tokenizer
from embedding import Embedding
from encoder import Encoder
from positional_encoding import PositionalEncoding

corpus = [
    "I love dogs.",
    "Dogs bark.",
    "Cats meow."
]

token = Tokenizer()

embedding = Embedding(5000, 128)

encoder = Encoder(input_dim=128, hidden_dim=512)

position = PositionalEncoding()


token.fit(corpus)
input_ids = token.brew('I LOVE DOGS', 10)["input_ids"]
input_weights = embedding.forward(input_ids)
positional_encoding = position.forward(input_weights.shape[0], input_weights.shape[1])
X = positional_encoding + input_weights
X = encoder.forward(X)
print(X.shape)
X = encoder.forward(X)
print(X.shape)
X = encoder.forward(X)
print(X.shape)

##------------------- Encoder can stack upon each other and the output of one encoder can be the input of another encoder. Providing the same shape of input and output. -------------------##