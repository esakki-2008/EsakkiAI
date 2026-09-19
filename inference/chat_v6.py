import re
import torch
import torch.nn.functional as F
import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from model.transformer import EsakkiGPT
from tokenizer.tokenizer import BPETokenizer


CHECKPOINT_PATH = "checkpoints/esakkiai_v6_best.pt"

DEVICE = torch.device("cpu")

# V6 architecture
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 4
FF_DIM = 512
MAX_SEQ_LEN = 128

TEMPERATURE = 0.70
TOP_K = 40
TOP_P = 0.90
REPETITION_PENALTY = 1.15
MAX_NEW_TOKENS = 80


def load_model():

    print("Loading EsakkiAI V6...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    print("Checkpoint loaded.")

    # -----------------------------------------
    # Read tokenizer information
    # -----------------------------------------

    if "stoi" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain 'stoi'. "
            "Please show the checkpoint keys."
        )

    if "itos" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain 'itos'. "
            "Please show the checkpoint keys."
        )

    if "merges" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain 'merges'. "
            "Please show the checkpoint keys."
        )

    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]
    merges = checkpoint["merges"]

    vocab_size = len(stoi)

    print(f"Vocabulary size: {vocab_size}")
    print(f"Model architecture: {D_MODEL}d / {N_LAYERS} layers")
    print(f"Heads: {N_HEADS}")
    print(f"FF dimension: {FF_DIM}")
    print(f"Context length: {MAX_SEQ_LEN}")

    # -----------------------------------------
    # Reconstruct tokenizer
    # -----------------------------------------

    tokenizer = BPETokenizer(
        vocab_size=vocab_size
    )

    tokenizer.stoi = stoi
    tokenizer.itos = itos
    tokenizer.merges = merges

    # -----------------------------------------
    # Reconstruct V6 model
    # -----------------------------------------

    model = EsakkiGPT(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        ff_dim=FF_DIM,
        max_seq_len=MAX_SEQ_LEN
    )

    # -----------------------------------------
    # Load weights
    # -----------------------------------------

    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    elif "model" in checkpoint:

        model.load_state_dict(
            checkpoint["model"]
        )

    elif "state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["state_dict"]
        )

    else:

        raise KeyError(
            "Could not find model weights in checkpoint."
        )

    model.to(DEVICE)
    model.eval()

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(f"Parameters: {parameters:,}")
    print("Model loaded successfully.")
    print()

    return model, tokenizer


def apply_repetition_penalty(
    logits,
    input_ids,
    penalty
):

    if penalty == 1.0:
        return logits

    unique_tokens = set(
        input_ids[0].tolist()
    )

    for token_id in unique_tokens:

        if logits[0, token_id] < 0:

            logits[0, token_id] *= penalty

        else:

            logits[0, token_id] /= penalty

    return logits


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

    logits = torch.full_like(
        logits,
        float("-inf")
    )

    logits.scatter_(
        1,
        sorted_indices,
        sorted_logits
    )

    return logits


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt
):

    full_prompt = (
        f"User: {prompt} EsakkiAI:"
    )

    input_ids = tokenizer.encode(
        full_prompt
    )

    if not input_ids:
        return ""

    input_tensor = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=DEVICE
    )

    generated = input_tensor.clone()

    initial_length = generated.shape[1]

    for _ in range(MAX_NEW_TOKENS):

        context = generated[
            :,
            -model.max_seq_len:
        ]

        logits = model(context)

        next_token_logits = logits[
            :,
            -1,
            :
        ]

        next_token_logits = (
            next_token_logits /
            TEMPERATURE
        )

        next_token_logits = apply_repetition_penalty(
            next_token_logits,
            context,
            REPETITION_PENALTY
        )

        next_token_logits = top_k_filter(
            next_token_logits,
            TOP_K
        )

        next_token_logits = top_p_filter(
            next_token_logits,
            TOP_P
        )

        probabilities = F.softmax(
            next_token_logits,
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

        new_tokens = generated[
            :,
            initial_length:
        ]

        text = tokenizer.decode(
            new_tokens[0].tolist()
        )

        # Stop when model starts another conversation turn.
        if re.search(
            r"\s+(User|EsakkiAI):",
            text
        ):
            break

    answer = tokenizer.decode(
        generated[
            0,
            initial_length:
        ].tolist()
    )

    # Remove next conversation turn.
    match = re.search(
        r"\s+(User|EsakkiAI):",
        answer
    )

    if match:
        answer = answer[
            :match.start()
        ]

    return answer.strip()


def main():

    model, tokenizer = load_model()

    print("=" * 60)
    print("EsakkiAI V6 Chat")
    print("=" * 60)
    print("Type 'exit' to quit.")
    print()

    test_questions = [
        "What is machine learning?",
        "What is a transformer?",
        "What is Python?",
        "What is artificial intelligence?",
        "What is a neural network?"
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