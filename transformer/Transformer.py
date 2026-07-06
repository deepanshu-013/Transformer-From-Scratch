from tokenizer import Tokenizer
from embedding import Embedding
from encoder import Encoder

corpus = [
    "I love dogs.",
    "Dogs bark.",
    "Cats meow."
]

token = Tokenizer()

embedding = Embedding(5000, 128)

encoder = Encoder(input_dim=128, hidden_dim=512)



token.fit(corpus)
input_ids = token.brew('I LOVE DOGS', 10)["input_ids"]
input_weights = embedding.forward(input_ids)
X = encoder.forward(input_weights)


print(X)
