import os
import sys
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

DEVICE = torch.device("cpu")

TEMPERATURE = 0.65
TOP_K = 30
TOP_P = 0.90
REPETITION_PENALTY = 1.10

MAX_NEW_TOKENS = 80


# ============================================================
# Load model
# ============================================================

def load_model():

    print("Loading EsakkiAI V7...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    vocab_size = checkpoint["vocab_size"]

    d_model = checkpoint["d_model"]
    n_heads = checkpoint["n_heads"]
    n_layers = checkpoint["n_layers"]
    ff_dim = checkpoint["ff_dim"]
    max_seq_len = checkpoint["max_seq_len"]

    tokenizer = V7Tokenizer(
        vocab_size=vocab_size
    )

    tokenizer.stoi = checkpoint["stoi"]
    tokenizer.itos = checkpoint["itos"]

    model = EsakkiGPT(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        ff_dim=ff_dim,
        max_seq_len=max_seq_len
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

    print("Checkpoint loaded.")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Parameters: {parameter_count:,}")
    print(f"Context length: {max_seq_len}")
    print(f"Best training step: {checkpoint['step']}")
    print(f"Best validation loss: {checkpoint['val_loss']:.4f}")
    print("Model loaded successfully.")
    print()

    return model, tokenizer


# ============================================================
# Repetition penalty
# ============================================================

def apply_repetition_penalty(
    logits,
    input_ids,
    penalty
):

    if penalty == 1.0:
        return logits

    token_ids = set(
        input_ids[0].tolist()
    )

    for token_id in token_ids:

        if logits[0, token_id] < 0:
            logits[0, token_id] *= penalty
        else:
            logits[0, token_id] /= penalty

    return logits


# ============================================================
# Top-K
# ============================================================

def top_k_filter(logits, k):

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

    threshold = values[:, -1].unsqueeze(-1)

    logits = torch.where(
        logits < threshold,
        torch.full_like(
            logits,
            float("-inf")
        ),
        logits
    )

    return logits


# ============================================================
# Top-P
# ============================================================

def top_p_filter(logits, p):

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

    filtered_logits = torch.full_like(
        logits,
        float("-inf")
    )

    filtered_logits.scatter_(
        1,
        sorted_indices,
        sorted_logits
    )

    return filtered_logits


# ============================================================
# Generate response
# ============================================================

@torch.no_grad()
def generate(
    model,
    tokenizer,
    question
):

    prompt = (
        "<USER>\n"
        + question.strip()
        + "\n<ASSISTANT>\n"
    )

    prompt_ids = tokenizer.encode(
        prompt
    )

    if not prompt_ids:
        return ""

    generated = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=DEVICE
    )

    initial_length = generated.shape[1]

    end_token_id = tokenizer.stoi.get(
        "<END>"
    )

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
            next_logits / TEMPERATURE
        )

        next_logits = apply_repetition_penalty(
            next_logits,
            context,
            REPETITION_PENALTY
        )

        next_logits = top_k_filter(
            next_logits,
            TOP_K
        )

        next_logits = top_p_filter(
            next_logits,
            TOP_P
        )

        probabilities = F.softmax(
            next_logits,
            dim=-1
        )

        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        )

        generated = torch.cat(
            [
                generated,
                next_token
            ],
            dim=1
        )

        if (
            end_token_id is not None
            and next_token.item() == end_token_id
        ):
            break

    response_ids = generated[
        0,
        initial_length:
    ].tolist()

    # Remove <END> from final answer.
    if end_token_id is not None:
        if end_token_id in response_ids:
            response_ids = response_ids[
                :response_ids.index(end_token_id)
            ]

    response = tokenizer.decode(
        response_ids
    )

    # Safety cleanup.
    response = response.replace(
        "<USER>",
        ""
    )

    response = response.replace(
        "<ASSISTANT>",
        ""
    )

    response = response.replace(
        "<END>",
        ""
    )

    return response.strip()


# ============================================================
# Main
# ============================================================

def main():

    model, tokenizer = load_model()

    print("=" * 60)
    print("EsakkiAI V7 Chat")
    print("=" * 60)
    print("Type 'exit' to quit.")
    print()

    test_questions = [
        "What is machine learning?",
        "What is a transformer?",
        "What is Python?",
        "What is artificial intelligence?",
        "What is a neural network?",
        "What is GitHub?",
        "What is caching?",
        "What is EsakkiAI?"
    ]

    print("Running automatic tests...")
    print()

    for question in test_questions:

        answer = generate(
            model,
            tokenizer,
            question
        )

        print(
            f"User: {question}"
        )

        print(
            f"EsakkiAI: {answer}"
        )

        print("-" * 60)

    print()
    print("Interactive mode")
    print()

    while True:

        try:

            question = input(
                "You: "
            ).strip()

            if question.lower() in {
                "exit",
                "quit",
                "q"
            }:

                print(
                    "EsakkiAI: Goodbye!"
                )

                break

            if not question:
                continue

            answer = generate(
                model,
                tokenizer,
                question
            )

            print(
                f"EsakkiAI: {answer}"
            )

            print()

        except KeyboardInterrupt:

            print(
                "\nEsakkiAI: Goodbye!"
            )

            break


if __name__ == "__main__":
    main()