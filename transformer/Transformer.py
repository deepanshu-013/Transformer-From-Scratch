from tokenizer import Tokenizer
from embedding import Embedding
from encoder import Encoder
from positional_encoding import PositionalEncoding
from lm_head import LMHead
from mlm_dataset import MLMDataset
from loss import CrossEntropyLoss



class Transformer:
    def __init__(self, vocab_size, d_model = 256, num_layers = 1):
        self.mlm_dataset = MLMDataset()
        self.embedding = Embedding(vocab_size, self.d_model)
        self.positional_encoding = PositionalEncoding()

        # Create Multiple Encoder as per the given value of num_layers
        self.encoders =  [Encoder(input_dim= d_model, hidden_dim=1024) for _ in range(num_layers)]

        self.lm_head = LMHead(vocab_size, d_model)

    def forward(self, X):
        # Embedding layer forward pass
        X = self.embedding.forward(X)

        # Positional encoding
        positional_encoding = self.positional_encoding.forward(X)

        # Adding the positional encoding with the embedding.
        X = X + positional_encoding

        # Encoders Layer
        for encoder in self.encoders:
            X = encoder.forward(X)

        # LM Head
        X =self.lm_head.forward(X)

        return X

    def backward(self, d_output):
        # LM Head backward pass
        dX = self.lm_head.backward(d_output)

        # Encoders backward
        for encoder in reversed(self.encoders):
            dX = encoder.backward(dX)

        # Embedding backward
        return self.embedding.backward(dX)

corpus = [
    "I love dogs.",
    "Dogs bark.",
    "Cats meow."
]

tokenizer = Tokenizer()
tokenizer.fit(corpus)

vocab_size = len(tokenizer.word_to_id)

transformer = Transformer(vocab_size, d_model= 256, num_layers= 4)

ce_loss = CrossEntropyLoss()

# Since brew returns a dictionary.
dictionary = tokenizer.brew("I love dogs.", max_length=10)

X, labels = transformer.mlm_dataset.prepare(dictionary[0])

X = transformer.forward(X)

loss = ce_loss.forward(X, labels)

d_logits = ce_loss.backward()

transformer.backward(d_logits)

