# EsakkiAI 🧠

> A GPT-style language model built from scratch for learning, experimentation, and understanding how language models work.

EsakkiAI is a lightweight educational language model built from the ground up using Python and PyTorch. The project implements its own tokenizer, Transformer architecture, training pipeline, evaluation system, inference engine, and conversational web interface.

## 🚀 V7 — Current Stable Version

EsakkiAI V7 is the current stable experimental model.

| Specification | V7 |
|---|---:|
| Parameters | **901,734 (~0.9M)** |
| Vocabulary | **358 tokens** |
| Context length | **128 tokens** |
| Transformer layers | **4** |
| Attention heads | **4** |
| Model dimension | **128** |
| Feed-forward dimension | **512** |
| Best validation loss | **0.1007** |
| Validation perplexity | **~1.11** |

V7 was trained with a custom PyTorch pipeline and evaluated using validation loss and generation tests.

## ✨ What EsakkiAI Implements

- Custom BPE-style tokenizer
- GPT-style decoder-only Transformer
- Causal self-attention
- Multi-head attention
- Token and positional embeddings
- Layer normalization
- Residual connections
- Feed-forward networks
- Next-token prediction
- Conversational training format
- Validation and perplexity evaluation
- Temperature, Top-K and Top-P sampling
- Repetition penalty
- Interactive conversational inference
- Flask-based web interface
- CPU inference
- No external AI API required for model generation

## 🧠 V7 Architecture

User → Web Interface → Flask API → V7 Tokenizer → Token Embeddings → Positional Embeddings → Transformer Blocks → Output Projection → Next-Token Prediction → Response

Transformer blocks use LayerNorm, multi-head causal self-attention, residual connections, and feed-forward networks.

## 🔤 Conversation Format

V7 uses special tokens including <PAD>, <UNK>, <USER>, <ASSISTANT>, and <END> to structure conversational training examples.

## 📚 Training

The V7 pipeline includes dataset generation, tokenization, train/validation splitting, mini-batch training, AdamW optimization, weight decay, gradient clipping, cosine learning-rate scheduling, periodic validation, best-checkpoint saving, and early-stopping logic.

## 📊 Evaluation

V7 was tested on Machine Learning, Artificial Intelligence, Neural Networks, Transformers, Python, Git and GitHub, Databases, Caching, Deep Learning, APIs, and EsakkiAI.

Evaluation:

- Validation loss: **0.1043**
- Perplexity: **1.11**

Generation tests showed strong performance on questions similar to the training distribution, while broader generalization remains a future improvement area.

## 💬 Web Interface

The project includes a Flask-based chat interface with message bubbles, suggested questions, typing feedback, responsive layout, and V7 model information.

## 🛠️ Tech Stack

**Python · PyTorch · NumPy · Flask · Git · GitHub**

## 💻 Development Environment

The project was developed and tested on a consumer laptop using CPU-based PyTorch training. The model was intentionally kept lightweight so the complete ML pipeline could be explored without expensive hardware.

## 🔒 Model Checkpoints

Large trained checkpoints are intentionally excluded from Git through .gitignore so the source repository remains lightweight.

## 💰 Cost

**Development cost: ₹0** using open-source tools and free resources.

## ⚠️ Current Limitations

EsakkiAI V7 is an educational small language model, not a general-purpose foundation model. Its knowledge is limited to its training distribution, it does not automatically know current events, and it can produce incorrect or incomplete responses.

## 🔭 Future Development

Future work may explore better datasets, improved tokenization, longer context, more efficient training, better conversational learning, retrieval-augmented generation, tool integration, improved evaluation, and more efficient architectures. Development will remain incremental rather than making a large jump in model size.

## 🎯 Learning Goals

This project is focused on understanding tokenization, embeddings, attention mechanisms, Transformer architecture, next-token prediction, dataset preparation, optimization, validation, perplexity, text generation, and inference systems.

## 👨‍💻 Author

**Esakki Raja**

GitHub: https://github.com/esakki-2008

Project: https://github.com/esakki-2008/EsakkiAI

## 📄 License

This project is intended for educational and experimental purposes.
