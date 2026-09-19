import os
import sys
import time

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

from tokenizer.tokenizer import BPETokenizer


# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = (
    "data/conversation_train.txt"
)

CHECKPOINT_PATH = (
    "checkpoints/esakkiai_v6_best.pt"
)

DEVICE = torch.device("cpu")


MAX_STEPS = 4000

BATCH_SIZE = 8

SEQ_LEN = 128

LEARNING_RATE = 0.0003

MIN_LEARNING_RATE = 0.00003

TOKENIZER_VOCAB_SIZE = 1000

TRAIN_SPLIT = 0.9

VALIDATION_INTERVAL = 100

PATIENCE = 7

GRAD_CLIP = 1.0


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 60)
print("EsakkiAI Stage 23 - v6 Training")
print("=" * 60)

print()

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
# TOKENIZER
# ============================================================

print()
print("Training BPE tokenizer...")


tokenizer = BPETokenizer(
    vocab_size=TOKENIZER_VOCAB_SIZE
)

tokenizer.train(text)


vocab_size = tokenizer.vocabulary_size


print(
    "Vocabulary:",
    vocab_size
)

print(
    "BPE merges:",
    len(tokenizer.merges)
)


# ============================================================
# ENCODE
# ============================================================

print()
print("Encoding dataset...")


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
# SPLIT
# ============================================================

split_index = int(
    len(data) * TRAIN_SPLIT
)


train_data = data[
    :split_index
]


val_data = data[
    split_index:
]


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
    d_model=128,
    n_heads=4,
    n_layers=4,
    ff_dim=512,
    max_seq_len=SEQ_LEN
)


model.to(DEVICE)


parameters = sum(
    p.numel()
    for p in model.parameters()
)


print()
print(
    "Model parameters:",
    f"{parameters:,}"
)

print(
    "Context length:",
    SEQ_LEN
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
# BATCH
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

def validation_loss():

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
# TRAIN
# ============================================================

print()
print("Starting v6 training...")
print()


start_time = time.time()

best_loss = float("inf")

no_improvement = 0


model.train()


for step in range(
    1,
    MAX_STEPS + 1
):

    x, y = get_batch(
        train_data
    )


    logits = model(x)


    loss = loss_function(
        logits.reshape(
            -1,
            vocab_size
        ),
        y.reshape(-1)
    )


    optimizer.zero_grad(
        set_to_none=True
    )


    loss.backward()


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

        val_loss = validation_loss()

        elapsed = (
            time.time()
            - start_time
        )

        lr = (
            optimizer.param_groups[0]["lr"]
        )


        print(
            f"Step {step:04d}/{MAX_STEPS} | "
            f"Train Loss: {loss.item():.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"LR: {lr:.6f} | "
            f"Time: {elapsed:.1f}s"
        )


        # ----------------------------------------------------
        # BEST
        # ----------------------------------------------------

        if val_loss < best_loss:

            best_loss = val_loss

            no_improvement = 0


            checkpoint = {

                "model_state_dict":
                    model.state_dict(),

                "vocab_size":
                    vocab_size,

                "d_model":
                    128,

                "n_heads":
                    4,

                "n_layers":
                    4,

                "ff_dim":
                    512,

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
                    best_loss,

                "step":
                    step,

                "version":
                    "v6",

                "dataset":
                    "conversation_train.txt"
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
                f"  Saved best v6 checkpoint "
                f"(Val Loss: {best_loss:.4f})"
            )


        else:

            no_improvement += 1


        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if no_improvement >= PATIENCE:

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
    "Stage 23 v6 training complete!"
)

print(
    "Best validation loss:",
    f"{best_loss:.4f}"
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