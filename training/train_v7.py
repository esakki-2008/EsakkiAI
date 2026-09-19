import os
import sys
import math
import time
import random

import torch
import torch.nn as nn

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from model.transformer import EsakkiGPT
from tokenizer.v7_tokenizer import V7Tokenizer


# ============================================================
# Configuration
# ============================================================

DATA_PATH = "data/conversation_v7.txt"
CHECKPOINT_PATH = "checkpoints/esakkiai_v7_best.pt"

DEVICE = torch.device("cpu")

VOCAB_SIZE = 1000

D_MODEL = 128
N_HEADS = 4
N_LAYERS = 4
FF_DIM = 512
MAX_SEQ_LEN = 128

BATCH_SIZE = 8
SEQ_LEN = 128

MAX_STEPS = 4000

LEARNING_RATE = 0.0003
MIN_LEARNING_RATE = 0.00003

WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

VALIDATION_INTERVAL = 100
VALIDATION_BATCHES = 30

PATIENCE = 10

TRAIN_SPLIT = 0.90

SEED = 42


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Load dataset
# ============================================================

print("=" * 60)
print("EsakkiAI V7 Training")
print("=" * 60)

print()
print("Loading dataset...")

with open(
    DATA_PATH,
    "r",
    encoding="utf-8"
) as f:
    text = f.read()

print(f"Characters: {len(text):,}")


# ============================================================
# Tokenizer
# ============================================================

print()
print("Building V7 tokenizer...")

tokenizer = V7Tokenizer(
    vocab_size=VOCAB_SIZE
)

actual_vocab_size = tokenizer.build_vocab(
    text
)

print(f"Vocabulary size: {actual_vocab_size}")


# ============================================================
# Encode dataset
# ============================================================

print()
print("Encoding dataset...")

tokens = tokenizer.encode(text)

print(f"Total tokens: {len(tokens):,}")


data = torch.tensor(
    tokens,
    dtype=torch.long
)


# ============================================================
# Train / validation split
# ============================================================

split_index = int(
    len(data) * TRAIN_SPLIT
)

train_data = data[:split_index]
val_data = data[split_index:]

print(
    f"Train tokens: {len(train_data):,}"
)

print(
    f"Validation tokens: {len(val_data):,}"
)


# ============================================================
# Model
# ============================================================

model = EsakkiGPT(
    vocab_size=actual_vocab_size,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    ff_dim=FF_DIM,
    max_seq_len=MAX_SEQ_LEN
).to(DEVICE)


parameter_count = sum(
    p.numel()
    for p in model.parameters()
)


print()
print(f"Model parameters: {parameter_count:,}")
print(f"Context length: {MAX_SEQ_LEN}")
print(f"Device: {DEVICE}")


# ============================================================
# Batch generation
# ============================================================

def get_batch(source):

    max_start = len(source) - SEQ_LEN - 1

    starts = torch.randint(
        0,
        max_start,
        (BATCH_SIZE,)
    )

    x = torch.stack([
        source[
            start:start + SEQ_LEN
        ]
        for start in starts
    ])

    y = torch.stack([
        source[
            start + 1:
            start + SEQ_LEN + 1
        ]
        for start in starts
    ])

    return (
        x.to(DEVICE),
        y.to(DEVICE)
    )


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def estimate_loss():

    model.eval()

    losses = []

    for _ in range(VALIDATION_BATCHES):

        x, y = get_batch(val_data)

        logits = model(x)

        loss = nn.functional.cross_entropy(
            logits.reshape(-1, actual_vocab_size),
            y.reshape(-1)
        )

        losses.append(
            loss.item()
        )

    model.train()

    return sum(losses) / len(losses)


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=MAX_STEPS,
    eta_min=MIN_LEARNING_RATE
)


# ============================================================
# Training
# ============================================================

print()
print("Starting training...")
print()

best_val_loss = float("inf")
best_step = 0
patience_counter = 0

start_time = time.time()


for step in range(1, MAX_STEPS + 1):

    model.train()

    x, y = get_batch(train_data)

    logits = model(x)

    loss = nn.functional.cross_entropy(
        logits.reshape(-1, actual_vocab_size),
        y.reshape(-1)
    )

    optimizer.zero_grad()

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        GRAD_CLIP
    )

    optimizer.step()

    scheduler.step()


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if (
        step == 1
        or step % VALIDATION_INTERVAL == 0
    ):

        val_loss = estimate_loss()

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Step {step:4d} | "
            f"Train Loss {loss.item():.4f} | "
            f"Val Loss {val_loss:.4f} | "
            f"LR {current_lr:.7f}"
        )


        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_step = step
            patience_counter = 0

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "stoi": tokenizer.stoi,
                "itos": tokenizer.itos,
                "merges": tokenizer.merges,

                "vocab_size": actual_vocab_size,

                "d_model": D_MODEL,
                "n_heads": N_HEADS,
                "n_layers": N_LAYERS,
                "ff_dim": FF_DIM,
                "max_seq_len": MAX_SEQ_LEN,

                "step": step,
                "val_loss": val_loss
            }

            torch.save(
                checkpoint,
                CHECKPOINT_PATH
            )

            print(
                f"  ✓ Best checkpoint saved "
                f"(step {step})"
            )

        else:

            patience_counter += 1

            if patience_counter >= PATIENCE:

                print()
                print(
                    "Early stopping triggered."
                )

                break


# ============================================================
# Finished
# ============================================================

elapsed = time.time() - start_time

print()
print("=" * 60)
print("V7 TRAINING COMPLETE")
print("=" * 60)

print(
    f"Best validation loss: "
    f"{best_val_loss:.4f}"
)

print(
    f"Best step: {best_step}"
)

print(
    f"Training time: "
    f"{elapsed:.1f} seconds"
)

print(
    f"Checkpoint: "
    f"{CHECKPOINT_PATH}"
)

print("=" * 60)