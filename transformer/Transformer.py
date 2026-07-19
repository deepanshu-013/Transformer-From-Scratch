from tokenizer import Tokenizer
from embedding import Embedding
from encoder import Encoder
from positional_encoding import PositionalEncoding
from lm_head import LMHead
from mlm_dataset import MLMDataset
from loss import CrossEntropyLoss
import numpy as np
from adam import Adam
from data import wikitext_2

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

# WIKITEXT-2 IMPORTED
corpus = wikitext_2.DataSet.wiki_text_2()

import pickle
with open("wikitext2_corpus.pkl", "wb") as f:
    pickle.dump(corpus, f)

# FIT TOKENIZER
tokenizer = Tokenizer()
tokenizer.fit(corpus)
vocab_size = len(tokenizer.word_to_id)
print(f"Vocabulary Size: {vocab_size}")

# INITIALIZE MODEL & OPTIMIZER
transformer = Transformer(vocab_size, d_model= 128, num_layers= 4)
ce_loss = CrossEntropyLoss()
optimizer = Adam(model=transformer, learning_rate=0.001, clip_value=1.0)

processed_corpus = []
for sentence in corpus:
    try:
        processed_corpus.append(tokenizer.brew(sentence, max_length= 64))
    except ValueError:
        continue

total_loss = []
train_data = [(s["input_ids"], s["attention_mask"]) for s in processed_corpus]
num_samples = len(train_data)

# Training Loop
for epoch in range(100):
    epoch_loss = 0.0
    correct_mask_predictions = 0
    total_mask_predictions = 0

    np.random.shuffle(train_data)

    for input_ids, attn_mask in train_data:
        # 1. Zero gradients (Handled by Adam now)
        optimizer.zero_grad()

        # 2. Forward pass
        X, labels = transformer.mlm_dataset.prepare(input_ids, tokenizer)

        # 3. Calculate Loss
        logits = transformer.forward(X, attn_mask)
        loss = ce_loss.forward(logits, labels)

        # 4. Backward pass
        d_logits = ce_loss.backward()
        transformer.backward(d_logits)

        # 5. Update weights (Handled by Adam now)
        optimizer.step()

        epoch_loss += loss

        # Accuracy calculation
        valid_mask = (labels != -100)
        if np.any(valid_mask):
            predictions = np.argmax(logits, axis=-1)
            total_mask_predictions += np.sum(valid_mask)
            correct_mask_predictions += np.sum(predictions[valid_mask] == labels[valid_mask])

    avg_loss = epoch_loss / num_samples
    accuracy = correct_mask_predictions / max(1, total_mask_predictions)

    # Print Gradients & Metrics
    if epoch % 2 == 0:
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

    print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Perplexity: {np.exp(avg_loss):.3f} | Acc: {accuracy:.4f}")

