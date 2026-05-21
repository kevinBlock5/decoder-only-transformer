# Decoder-Only Transformer From Scratch

Implementation of a GPT-style decoder-only transformer architecture from scratch using PyTorch.

---

## 🚀 Overview

This project focuses on understanding the internal architecture of decoder-only transformers by manually implementing the core building blocks using PyTorch.

The implementation includes:
- Token embeddings
- Positional encoding
- Multi-head self-attention
- Feed-forward neural networks
- Decoder blocks
- Residual connections
- Layer normalization
- Output projection head

---

## 🧠 Architecture Flow

Input Tokens
      ↓
Embedding Layer
      ↓
Positional Encoding
      ↓
Decoder Stack
  ├── Multi-Head Self Attention
  ├── Residual Connection
  ├── Feed Forward Network
  └── Layer Normalization
      ↓
Linear Head
      ↓
Token Predictions

---

## ✨ Features

- Decoder-only transformer architecture
- Multi-head self-attention implementation
- Positional encoding
- Residual connections
- Layer normalization
- Feed-forward network
- Modular PyTorch implementation
- Causal masking support

---

## 🛠 Tech Stack

- Python
- PyTorch

---

## 📂 Project Structure

```text
decoder-only-transformer/
│
├── model.py
├── README.md
└── requirements.txt
```

---

## 🎯 Goal

The goal of this project is to deeply understand:
- Transformer internals
- Attention mechanisms
- Decoder architectures
- Sequence modeling
- Deep learning system design

through manual implementation instead of relying only on high-level APIs.

---

## 📚 Concepts Explored

- Self-Attention
- Multi-Head Attention
- Positional Encoding
- Residual Connections
- Layer Normalization
- Feed Forward Networks
- Autoregressive Modeling

---

## 🔮 Future Improvements

- Training pipeline
- Tokenizer integration
- Text generation
- Sampling strategies
- KV caching
- Training on custom datasets
- Performance optimization

---

## ⚡ Status

Learning-focused implementation / work in progress.
