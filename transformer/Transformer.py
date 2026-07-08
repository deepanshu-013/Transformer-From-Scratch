from tokenizer import Tokenizer
from embedding import Embedding
from encoder import Encoder
from positional_encoding import PositionalEncoding
from lm_head import LMHead

class Transformer:
    def __init__(self, num_layers = 2):
        self.encoders = [
            Encoder() for _ in range(num_layers)
        ]

    def forward(self, X):
        for encoder in self.encoders:
            X = encoder.forward(X)
        return X

corpus = [
    "I love dogs.",
    "Dogs bark.",
    "Cats meow."
]

token = Tokenizer()

embedding = Embedding(5000, 128)

encoder = Encoder(input_dim=128, hidden_dim=512)

position = PositionalEncoding()

transformer = Transformer(num_layers=2)

lm_head = LMHead(vocab_size=5000, embedding_dim=128)

token.fit(corpus)
input_ids = token.brew('I LOVE DOGS', 10)["input_ids"]
input_weights = embedding.forward(input_ids)
positional_encoding = position.forward(input_weights.shape[0], input_weights.shape[1])
X = positional_encoding + input_weights
X = transformer.forward(X)
X = lm_head.forward(X) 
print(X.shape)


