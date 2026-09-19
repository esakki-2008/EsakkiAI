# EsakkiAI 🧠

> A GPT-style language model built from scratch for learning, experimentation, and understanding how modern language models work.

EsakkiAI is a small educational language model built from the ground up using Python and PyTorch.

The project implements its own tokenizer, Transformer architecture, training pipeline, evaluation system, inference engine, and local web interface.

The goal is to understand the fundamental components behind GPT-style models rather than simply calling an external AI API.

---

## ✨ Features

- Custom tokenizer
- GPT-style decoder-only Transformer
- Causal self-attention
- Multi-head attention
- Positional embeddings
- Feed-forward layers
- Layer normalization
- Next-token prediction
- Custom training pipeline
- Conversational dataset
- Validation and perplexity evaluation
- Temperature sampling
- Top-K sampling
- Top-P sampling
- Repetition penalty
- Local CPU inference
- Flask web interface
- No external AI API required

---

## 🧠 Architecture

```text
User
 │
 ▼
Web Interface
 │
 ▼
Flask API
 │
 ▼
V7 Tokenizer
 │
 ▼
Token IDs
 │
 ▼
Token Embeddings
 │
 ▼
Positional Embeddings
 │
 ▼
Transformer Blocks
 │
 ├── LayerNorm
 ├── Multi-Head Self-Attention
 ├── Residual Connection
 ├── LayerNorm
 ├── Feed-Forward Network
 └── Residual Connection
 │
 ▼
Output Projection
 │
 ▼
Next Token Prediction
 │
 ▼
EsakkiAI Response