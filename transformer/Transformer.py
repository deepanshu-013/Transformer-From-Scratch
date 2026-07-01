from tokenizer import Tokenizer
from embedding import Embedding



token = Tokenizer()

corpus = [
    "I love dogs.",
    "Dogs bark.",
    "Cats meow."
]

token.fit(corpus)

embedding = Embedding(5000, 128)

print(token.encode('I LOVE DOGS.'))

print(embedding.forward(token.encode('I LOVE DOGS.')).shape)

print(token.decode(token.encode('I LOVE DOGS.')))
