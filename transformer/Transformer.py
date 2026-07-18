from tokenizer import Tokenizer
from embedding import Embedding
from encoder import Encoder
from positional_encoding import PositionalEncoding
from lm_head import LMHead
from mlm_dataset import MLMDataset
from loss import CrossEntropyLoss
from sgd import SGD
import numpy as np
import random as rn

class Transformer:
    def __init__(self, vocab_size, d_model = 256, num_layers = 1):
        self.mlm_dataset = MLMDataset()
        self.embedding = Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding()

        # Create Multiple Encoder as per the given value of num_layers
        self.encoders =  [Encoder(input_dim= d_model, hidden_dim=128) for _ in range(num_layers)]

        self.lm_head = LMHead(vocab_size, d_model)

    def forward(self, X, attention_mask):
        # Embedding layer forward pass
        X = self.embedding.forward(X)

        # Positional encoding
        positional_encoding = self.positional_encoding.forward(X)

        # Adding the positional encoding with the embedding.
        X = X + positional_encoding

        # Encoders Layer
        for encoder in self.encoders:
            X = encoder.forward(X, attention_mask)

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
        self.embedding.backward(dX)

    def update(self, learning_rate):
        self.lm_head.update(learning_rate)

        for encoder in self.encoders:
            encoder.update(learning_rate)

        self.embedding.update(learning_rate)

    def zero_grad(self):
        self.lm_head.zero_grad()

        self.embedding.zero_grad()

        for encoder in self.encoders:
            encoder.zero_grad()

    @staticmethod
    def train(input_ids, attention_mask):
        X, labels = transformer.mlm_dataset.prepare(input_ids)

        optimizer.zero_grad()

        logits = transformer.forward(X, attention_mask)

        loss = ce_loss.forward(logits, labels)

        d_logits = ce_loss.backward()

        transformer.backward(d_logits)

        optimizer.step()

        return X, logits, loss

corpus = [
    'I love dogs.',
    'Dogs bark loudly.',
    'Cats chase mice.',
    'Birds can fly.',
    'Fish swim underwater.',
    'The sun is bright.',
    'The moon shines at night.'
]

tokenizer = Tokenizer()
tokenizer.fit(corpus)

vocab_size = len(tokenizer.word_to_id)

transformer = Transformer(vocab_size, d_model= 64, num_layers= 1)

ce_loss = CrossEntropyLoss()

optimizer = SGD(model = transformer, learning_rate= 0.01)

total_loss = []

# Training Loop
for epoch in range(500):
    epoch_loss = 0
    rn.shuffle(corpus)

    for sentence in corpus:
        dictionary = tokenizer.brew(sentence, max_length=10)
        X, logits, loss = transformer.train(dictionary['input_ids'], dictionary['attention_mask'])
        total_loss.append(loss)
        epoch_loss += loss
    if epoch % 20 == 0:
        print("Embedding :", np.linalg.norm(transformer.embedding.d_weights))

        for i, encoder in enumerate(transformer.encoders):
            print(f"Encoder {i}")

            print("WQ :", np.linalg.norm(encoder.self_attention.dWQ))
            print("WK :", np.linalg.norm(encoder.self_attention.dWK))
            print("WV :", np.linalg.norm(encoder.self_attention.dWV))

            print("FF1:", np.linalg.norm(encoder.feedforward.dweights1))
            print("FF2:", np.linalg.norm(encoder.feedforward.dweights2))

        print("LM :", np.linalg.norm(transformer.lm_head.linear.d_weights))
        print(
            f"Epoch {epoch:3d}",
            f"loss: {epoch_loss / len(corpus):.3f}"
        )

