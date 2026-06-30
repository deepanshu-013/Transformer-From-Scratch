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

print(token.encode())

print(embedding.forward(token.encode()).shape)

print(token.decode(token.encode()))

print(token.word_to_id)