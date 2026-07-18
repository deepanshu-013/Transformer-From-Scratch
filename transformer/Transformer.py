from tokenizer import Tokenizer
from embedding import Embedding
from encoder import Encoder
from positional_encoding import PositionalEncoding
from lm_head import LMHead
from mlm_dataset import MLMDataset
from loss import CrossEntropyLoss
from sgd import SGD
import numpy as np
from synthetic_data import generate_synthetic_corpus

class Transformer:
    def __init__(self, vocab_size, d_model = 256, num_layers = 1):
        self.mlm_dataset = MLMDataset()
        self.embedding = Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding()

        # Create Multiple Encoder as per the given value of num_layers
        self.encoders =  [Encoder(input_dim= d_model, hidden_dim=512) for _ in range(num_layers)]

        self.lm_head = LMHead(self.embedding.weights)

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

        predictions = np.argmax(logits, axis=1)

        return X, predictions, labels, loss

def load_corpus(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

corpus = load_corpus("synthetic_corpus.txt")

tokenizer = Tokenizer()
tokenizer.fit(corpus)

vocab_size = len(tokenizer.word_to_id)

transformer = Transformer(vocab_size, d_model= 128, num_layers= 4)

ce_loss = CrossEntropyLoss()

optimizer = SGD(model = transformer, learning_rate= 0.01)

total_loss = []

# Debugging prints
print(f"Original corpus size: {len(corpus)}")

processed_corpus = []
for sentence in corpus:
    processed_corpus.append(
        tokenizer.brew(sentence, max_length=10)
    )

print(f"Processed corpus size: {len(processed_corpus)}")

# Pre-extract data
train_data = [(s["input_ids"], s["attention_mask"]) for s in processed_corpus]
num_samples = len(train_data)
print(f"Number of training samples: {num_samples}")

# ... then start the training loop ...

# Training Loop
for epoch in range(100):
    epoch_loss = 0.0
    correct_mask_predictions = 0
    total_mask_predictions = 0

    # 2. SHUFFLE DATA (Crucial for SGD to work properly on small datasets)
    np.random.shuffle(train_data)

    for input_ids, attn_mask in train_data:
        X, predictions, labels, loss = transformer.train(input_ids, attn_mask)

        epoch_loss += loss

        # 3. VECTORIZED ACCURACY CALCULATION (No Python for-loops over tokens)
        # This replaces your slow: for pred, label in zip(predictions, labels)...
        valid_mask = (labels != -100)
        if np.any(valid_mask):
            total_mask_predictions += np.sum(valid_mask)
            correct_mask_predictions += np.sum(predictions[valid_mask] == labels[valid_mask])

    # 4. CALCULATE METRICS ONCE PER EPOCH (Outside the inner loop)
    avg_loss = epoch_loss / num_samples if num_samples > 0 else 0.0
    accuracy = correct_mask_predictions / max(1, total_mask_predictions)  # max to avoid div by zero

    if epoch % 20 == 0:
        print("--- Gradients ---")
        print("Embedding :", np.linalg.norm(transformer.embedding.d_weights))
        for i, encoder in enumerate(transformer.encoders):
            print(f"Encoder {i}")
            print("WQ :", np.linalg.norm(encoder.self_attention.dWQ))
            print("WK :", np.linalg.norm(encoder.self_attention.dWK))
            print("WV :", np.linalg.norm(encoder.self_attention.dWV))
            print("FF1:", np.linalg.norm(encoder.feedforward.dweights1))
            print("FF2:", np.linalg.norm(encoder.feedforward.dweights2))
        print("LM :", np.linalg.norm(transformer.lm_head.linear.d_weights))

    # 5. CLEAN PRINT STATEMENT
    print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Perplexity: {np.exp(avg_loss):.3f} | Acc: {accuracy:.4f}")


import pickle

