# Transformer From Scratch (NumPy)

A complete implementation of the Transformer architecture built entirely from scratch using **NumPy**, without relying on deep learning frameworks such as PyTorch or TensorFlow.

This project was created as an educational implementation to understand every mathematical component of the Transformer, including forward propagation, backpropagation, optimization, and masked language model (MLM) training.

---

## Features

- Pure NumPy implementation
- Word-level tokenizer
- Token embeddings
- Sinusoidal positional encoding
- Multi-Head Self Attention
- Feed Forward Network
- Residual connections
- Layer Normalization
- Masked Language Modeling (MLM)
- Cross Entropy Loss
- Adam Optimizer implemented from scratch
- Gradient Clipping
- Dynamic Mask Generation
- WikiText-2 dataset support
- End-to-end training pipeline

---

## Project Structure

```
Transformer-From-Scratch/
│
├── data/
├── tokenizer.py
├── embedding.py
├── positional_encoding.py
├── attention.py
├── feedforward.py
├── layernorm.py
├── encoder.py
├── transformer.py
├── loss.py
├── adam.py
├── dataset.py
├── trainer.py
└── main.py
```

---

## What was implemented?

Instead of using automatic differentiation, every major component was implemented manually.

### Tokenizer

- Vocabulary creation
- Encoding / Decoding
- Special tokens
    - `[PAD]`
    - `[UNK]`
    - `[CLS]`
    - `[SEP]`
    - `[MASK]`

---

### Embedding Layer

- Learnable embedding matrix
- Forward propagation
- Manual gradient computation

---

### Positional Encoding

Implemented the original sinusoidal positional encoding introduced in the Transformer paper.

---

### Multi-Head Self Attention

Implemented manually:

- Query
- Key
- Value
- Scaled Dot Product Attention
- Softmax
- Attention masking
- Manual backward propagation

---

### Feed Forward Network

Two-layer MLP

```
Linear
↓
ReLU
↓
Linear
```

---

### Layer Normalization

Manual implementation of

- forward pass
- backward pass

without using automatic differentiation.

---

### Residual Connections

Residual connections are implemented throughout the encoder architecture.

---

### Optimizer

Custom implementation of Adam including

- Momentum
- Variance estimation
- Bias correction
- Gradient clipping

---

### Loss Function

Cross Entropy Loss with masked token support for MLM training.

---

### Training

The project supports training using

- Synthetic datasets
- WikiText-2

with dynamic masking during training.

---

## Why this project?

The goal of this repository was **learning**, not performance.

Modern frameworks hide most of the mathematics behind automatic differentiation.

This project exposes every matrix multiplication, every gradient, and every parameter update to better understand how Transformers actually work.

---

## Limitations

Since the project is implemented entirely using NumPy,

- training is CPU only
- training is significantly slower than GPU implementations
- not intended for large-scale language model pretraining

---

## Next Step

This repository serves as the foundation for a new implementation built using **PyTorch**.

The upcoming project focuses on

- GPU acceleration
- larger datasets
- better tokenizer
- larger models
- modern Transformer improvements

while preserving the understanding gained from implementing every component manually.

---

## References

- Attention Is All You Need
- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
- The Illustrated Transformer
- Dive into Deep Learning

---

## License

MIT License
