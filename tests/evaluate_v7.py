import os
import sys
import math
import re

import torch
import torch.nn.functional as F

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

CHECKPOINT_PATH = "checkpoints/esakkiai_v7_best.pt"
DATA_PATH = "data/conversation_v7.txt"

DEVICE = torch.device("cpu")

SEQ_LEN = 128
BATCH_SIZE = 8

VALIDATION_BATCHES = 100

TEMPERATURE = 0.65
TOP_K = 30
TOP_P = 0.90
REPETITION_PENALTY = 1.10

MAX_NEW_TOKENS = 80


# ============================================================
# Load checkpoint
# ============================================================

print("=" * 65)
print("EsakkiAI V7 Evaluation")
print("=" * 65)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)

vocab_size = checkpoint["vocab_size"]

tokenizer = V7Tokenizer(
    vocab_size=vocab_size
)

tokenizer.stoi = checkpoint["stoi"]
tokenizer.itos = checkpoint["itos"]


model = EsakkiGPT(
    vocab_size=vocab_size,
    d_model=checkpoint["d_model"],
    n_heads=checkpoint["n_heads"],
    n_layers=checkpoint["n_layers"],
    ff_dim=checkpoint["ff_dim"],
    max_seq_len=checkpoint["max_seq_len"]
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)
model.eval()


parameter_count = sum(
    p.numel()
    for p in model.parameters()
)


print()
print(f"Vocabulary size: {vocab_size}")
print(f"Parameters: {parameter_count:,}")
print(f"Best training step: {checkpoint['step']}")
print(
    f"Training best validation loss: "
    f"{checkpoint['val_loss']:.4f}"
)


# ============================================================
# Load dataset
# ============================================================

with open(
    DATA_PATH,
    "r",
    encoding="utf-8"
) as f:

    text = f.read()


tokens = tokenizer.encode(text)

data = torch.tensor(
    tokens,
    dtype=torch.long
)


split = int(
    len(data) * 0.90
)

val_data = data[split:]


print()
print(f"Total tokens: {len(data):,}")
print(f"Validation tokens: {len(val_data):,}")


# ============================================================
# Validation loss
# ============================================================

def get_validation_batch():

    max_start = (
        len(val_data)
        - SEQ_LEN
        - 1
    )

    starts = torch.randint(
        0,
        max_start,
        (BATCH_SIZE,)
    )

    x = torch.stack([
        val_data[
            start:start + SEQ_LEN
        ]
        for start in starts
    ])

    y = torch.stack([
        val_data[
            start + 1:
            start + SEQ_LEN + 1
        ]
        for start in starts
    ])

    return (
        x.to(DEVICE),
        y.to(DEVICE)
    )


print()
print("Calculating validation loss...")


losses = []

with torch.no_grad():

    for _ in range(VALIDATION_BATCHES):

        x, y = get_validation_batch()

        logits = model(x)

        loss = F.cross_entropy(
            logits.reshape(
                -1,
                vocab_size
            ),
            y.reshape(-1)
        )

        losses.append(
            loss.item()
        )


validation_loss = sum(losses) / len(losses)

perplexity = math.exp(
    validation_loss
)


print(
    f"Evaluation validation loss: "
    f"{validation_loss:.4f}"
)

print(
    f"Perplexity: "
    f"{perplexity:.2f}"
)


# ============================================================
# Sampling helpers
# ============================================================

def repetition_penalty(
    logits,
    input_ids,
    penalty
):

    if penalty == 1.0:
        return logits

    for token_id in set(
        input_ids[0].tolist()
    ):

        if logits[0, token_id] < 0:

            logits[0, token_id] *= penalty

        else:

            logits[0, token_id] /= penalty

    return logits


def top_k(logits, k):

    if k <= 0:
        return logits

    k = min(
        k,
        logits.size(-1)
    )

    values, _ = torch.topk(
        logits,
        k
    )

    threshold = values[
        :,
        -1
    ].unsqueeze(-1)

    return torch.where(
        logits < threshold,
        torch.full_like(
            logits,
            float("-inf")
        ),
        logits
    )


def top_p(logits, p):

    if p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(
        logits,
        descending=True
    )

    sorted_probs = torch.softmax(
        sorted_logits,
        dim=-1
    )

    cumulative_probs = torch.cumsum(
        sorted_probs,
        dim=-1
    )

    remove = cumulative_probs > p

    remove[:, 1:] = remove[:, :-1].clone()
    remove[:, 0] = False

    sorted_logits[remove] = float("-inf")

    filtered = torch.full_like(
        logits,
        float("-inf")
    )

    filtered.scatter_(
        1,
        sorted_indices,
        sorted_logits
    )

    return filtered


# ============================================================
# Generation
# ============================================================

@torch.no_grad()
def generate(question):

    prompt = (
        "<USER>\n"
        + question.strip()
        + "\n<ASSISTANT>\n"
    )

    prompt_ids = tokenizer.encode(
        prompt
    )

    generated = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=DEVICE
    )

    initial_length = generated.shape[1]

    end_id = tokenizer.stoi["<END>"]

    for _ in range(MAX_NEW_TOKENS):

        context = generated[
            :,
            -model.max_seq_len:
        ]

        logits = model(context)

        next_logits = logits[
            :,
            -1,
            :
        ]

        next_logits = (
            next_logits /
            TEMPERATURE
        )

        next_logits = repetition_penalty(
            next_logits,
            context,
            REPETITION_PENALTY
        )

        next_logits = top_k(
            next_logits,
            TOP_K
        )

        next_logits = top_p(
            next_logits,
            TOP_P
        )

        probabilities = torch.softmax(
            next_logits,
            dim=-1
        )

        next_token = torch.multinomial(
            probabilities,
            1
        )

        generated = torch.cat(
            [
                generated,
                next_token
            ],
            dim=1
        )

        if next_token.item() == end_id:
            break


    response_ids = generated[
        0,
        initial_length:
    ].tolist()


    if end_id in response_ids:

        response_ids = response_ids[
            :response_ids.index(end_id)
        ]


    response = tokenizer.decode(
        response_ids
    )

    response = response.replace(
        "<USER>",
        ""
    )

    response = response.replace(
        "<ASSISTANT>",
        ""
    )

    return response.strip()


# ============================================================
# Test questions
# ============================================================

test_questions = [

    "What is machine learning?",

    "What is a transformer?",

    "What is Python?",

    "What is artificial intelligence?",

    "What is a neural network?",

    "What is GitHub?",

    "What is caching?",

    "What is an API?",

    "What is deep learning?",

    "What is EsakkiAI?",

    # Unseen / paraphrased questions
    "Can you explain machine learning?",

    "Explain Python in simple terms.",

    "Why are transformers useful?",

    "What does Git do?",

    "Why is caching useful?"
]


# ============================================================
# Generation evaluation
# ============================================================

print()
print("=" * 65)
print("GENERATION TESTS")
print("=" * 65)

for index, question in enumerate(
    test_questions,
    start=1
):

    answer = generate(question)

    print()
    print(
        f"[{index}/{len(test_questions)}]"
    )

    print(
        f"User: {question}"
    )

    print(
        f"EsakkiAI: {answer}"
    )


# ============================================================
# Conversation boundary test
# ============================================================

print()
print("=" * 65)
print("CONVERSATION BOUNDARY TEST")
print("=" * 65)

boundary_question = (
    "What is Python?"
)

boundary_answer = generate(
    boundary_question
)

leakage_patterns = [
    "<USER>",
    "<ASSISTANT>",
    "<END>"
]

leakage_found = []

for pattern in leakage_patterns:

    if pattern in boundary_answer:

        leakage_found.append(
            pattern
        )


print()
print(
    f"Question: {boundary_question}"
)

print(
    f"Answer: {boundary_answer}"
)

if leakage_found:

    print()
    print(
        "Boundary leakage detected:",
        leakage_found
    )

else:

    print()
    print(
        "No conversation-token leakage detected."
    )


# ============================================================
# Final summary
# ============================================================

print()
print("=" * 65)
print("V7 EVALUATION COMPLETE")
print("=" * 65)

print(
    f"Validation loss: {validation_loss:.4f}"
)

print(
    f"Perplexity: {perplexity:.2f}"
)

print(
    f"Parameters: {parameter_count:,}"
)

print(
    f"Checkpoint: {CHECKPOINT_PATH}"
)

print("=" * 65)