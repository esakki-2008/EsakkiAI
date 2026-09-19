import os
import sys
import time

import torch
import torch.nn as nn


# ============================================================
# PROJECT PATH
# ============================================================

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from model.transformer import EsakkiGPT
from tokenizer.tokenizer import BPETokenizer


# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = "data/conversation_train.txt"

CHECKPOINT_PATH = (
    "checkpoints/esakkiai_v5_conversation_best.pt"
)

DEVICE = torch.device("cpu")

MAX_STEPS = 3000

BATCH_SIZE = 8

SEQ_LEN = 64

LEARNING_RATE = 0.0003

MIN_LEARNING_RATE = 0.00005

TOKENIZER_VOCAB_SIZE = 1000

TRAIN_SPLIT = 0.9

VALIDATION_INTERVAL = 100

PATIENCE = 5

GRAD_CLIP = 1.0


# ============================================================
# LOAD DATASET
# ============================================================

print()
print("=" * 60)
print("EsakkiAI Stage 22 Conversational Training")
print("=" * 60)

print()
print("Loading conversational dataset...")


with open(
    DATA_PATH,
    "r",
    encoding="utf-8"
) as file:

    text = file.read()


print(
    "Dataset characters:",
    len(text)
)


# ============================================================
# TRAIN BPE TOKENIZER
# ============================================================

print()
print("Training BPE tokenizer...")


tokenizer = BPETokenizer(
    vocab_size=TOKENIZER_VOCAB_SIZE
)

tokenizer.train(text)


vocab_size = tokenizer.vocabulary_size


print(
    "BPE vocabulary size:",
    vocab_size
)

print(
    "BPE merges:",
    len(tokenizer.merges)
)


# ============================================================
# ENCODE DATASET
# ============================================================

print()
print("Encoding conversational dataset...")


encoded = tokenizer.encode(
    text
)


data = torch.tensor(
    encoded,
    dtype=torch.long
)


print(
    "Total tokens:",
    len(data)
)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

split_index = int(
    len(data) * TRAIN_SPLIT
)


train_data = data[:split_index]

val_data = data[split_index:]


print(
    "Training tokens:",
    len(train_data)
)

print(
    "Validation tokens:",
    len(val_data)
)


# ============================================================
# MODEL
# ============================================================

model = EsakkiGPT(
    vocab_size=vocab_size,
    d_model=96,
    n_heads=4,
    n_layers=3,
    ff_dim=384,
    max_seq_len=SEQ_LEN
)


model = model.to(DEVICE)


parameters = sum(
    p.numel()
    for p in model.parameters()
)


print(
    "Model parameters:",
    f"{parameters:,}"
)

print(
    "Device:",
    DEVICE
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01
)


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=MAX_STEPS,
    eta_min=MIN_LEARNING_RATE
)


loss_function = nn.CrossEntropyLoss()


# ============================================================
# BATCH CREATION
# ============================================================

def get_batch(source):

    max_start = (
        len(source)
        - SEQ_LEN
        - 1
    )


    starts = torch.randint(
        0,
        max_start,
        (BATCH_SIZE,)
    )


    offsets = torch.arange(
        SEQ_LEN
    )


    x = source[
        starts[:, None] + offsets
    ]


    y = source[
        starts[:, None]
        + offsets
        + 1
    ]


    return (
        x.to(DEVICE),
        y.to(DEVICE)
    )


# ============================================================
# VALIDATION
# ============================================================

def calculate_validation_loss():

    model.eval()

    losses = []


    with torch.no_grad():

        for _ in range(30):

            x, y = get_batch(
                val_data
            )


            logits = model(x)


            loss = loss_function(
                logits.reshape(
                    -1,
                    vocab_size
                ),
                y.reshape(-1)
            )


            losses.append(
                loss.item()
            )


    model.train()


    return (
        sum(losses)
        / len(losses)
    )


# ============================================================
# TRAINING
# ============================================================

print()
print("Starting conversational training...")
print()


model.train()

start_time = time.time()

best_val_loss = float("inf")

steps_without_improvement = 0


for step in range(
    1,
    MAX_STEPS + 1
):

    x, y = get_batch(
        train_data
    )


    # --------------------------------------------------------
    # FORWARD
    # --------------------------------------------------------

    logits = model(x)


    loss = loss_function(
        logits.reshape(
            -1,
            vocab_size
        ),
        y.reshape(-1)
    )


    # --------------------------------------------------------
    # BACKWARD
    # --------------------------------------------------------

    optimizer.zero_grad(
        set_to_none=True
    )


    loss.backward()


    # --------------------------------------------------------
    # GRADIENT CLIPPING
    # --------------------------------------------------------

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        GRAD_CLIP
    )


    optimizer.step()

    scheduler.step()


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if (
        step == 1
        or step % VALIDATION_INTERVAL == 0
    ):

        val_loss = calculate_validation_loss()


        elapsed = (
            time.time()
            - start_time
        )


        current_lr = (
            optimizer.param_groups[0]["lr"]
        )


        print(
            f"Step {step:04d}/{MAX_STEPS} | "
            f"Train Loss: {loss.item():.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"Time: {elapsed:.1f}s"
        )


        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            steps_without_improvement = 0


            checkpoint = {

                "model_state_dict":
                    model.state_dict(),

                "vocab_size":
                    vocab_size,

                "d_model":
                    96,

                "n_heads":
                    4,

                "n_layers":
                    3,

                "ff_dim":
                    384,

                "max_seq_len":
                    SEQ_LEN,

                "stoi":
                    tokenizer.stoi,

                "itos":
                    tokenizer.itos,

                "merges":
                    tokenizer.merges,

                "tokenizer_vocab_size":
                    TOKENIZER_VOCAB_SIZE,

                "best_val_loss":
                    best_val_loss,

                "step":
                    step,

                "dataset":
                    "conversation_train.txt",

                "format":
                    "User: question / EsakkiAI: answer"
            }


            os.makedirs(
                "checkpoints",
                exist_ok=True
            )


            torch.save(
                checkpoint,
                CHECKPOINT_PATH
            )


            print(
                f"  Best conversational "
                f"checkpoint saved "
                f"(Val Loss: {best_val_loss:.4f})"
            )


        else:

            steps_without_improvement += 1


        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if (
            steps_without_improvement
            >= PATIENCE
        ):

            print()
            print(
                "Early stopping triggered."
            )

            break


# ============================================================
# COMPLETE
# ============================================================

elapsed = (
    time.time()
    - start_time
)


print()
print("=" * 60)

print(
    "Stage 22 conversational training complete!"
)

print(
    "Best validation loss:",
    f"{best_val_loss:.4f}"
)

print(
    "Training time:",
    f"{elapsed:.1f}s"
)

print()
print(
    "Checkpoint:"
)

print(
    CHECKPOINT_PATH
)

print("=" * 60)