import os
import sys
import re

import torch
import torch.nn.functional as F


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


# ============================================================
# SETTINGS
# ============================================================

CHECKPOINT_PATH = (
    "checkpoints/esakkiai_v5_conversation_best.pt"
)

DEVICE = torch.device("cpu")

MAX_NEW_TOKENS = 50

TEMPERATURE = 0.65

TOP_K = 30

TOP_P = 0.85

REPETITION_PENALTY = 1.20


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print()
print("Loading EsakkiAI conversational model...")


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)


stoi = checkpoint["stoi"]

itos = {
    int(k): v
    for k, v in checkpoint["itos"].items()
}

merges = [
    tuple(pair)
    for pair in checkpoint["merges"]
]

vocab_size = checkpoint["vocab_size"]


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


# ============================================================
# TOKENIZER
# ============================================================

class BPETokenizer:

    def __init__(
        self,
        stoi,
        itos,
        merges
    ):

        self.stoi = stoi

        self.itos = itos

        self.merges = merges


    def split_text(self, text):

        return re.findall(
            r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\w\s]|\s+",
            text
        )


    def apply_merges(self, pieces):

        for left, right in self.merges:

            merged = []

            i = 0

            while i < len(pieces):

                if (
                    i < len(pieces) - 1
                    and pieces[i] == left
                    and pieces[i + 1] == right
                ):

                    merged.append(
                        left + right
                    )

                    i += 2

                else:

                    merged.append(
                        pieces[i]
                    )

                    i += 1

            pieces = merged

        return pieces


    def encode(self, text):

        tokens = []

        for part in self.split_text(text):

            if part.isspace():

                pieces = [" "]

            else:

                pieces = self.apply_merges(
                    list(part)
                )


            for piece in pieces:

                if piece in self.stoi:

                    tokens.append(
                        self.stoi[piece]
                    )

                else:

                    for char in piece:

                        if char in self.stoi:

                            tokens.append(
                                self.stoi[char]
                            )

        return tokens


    def decode(self, tokens):

        return "".join(
            self.itos[token]
            for token in tokens
            if token in self.itos
        )


tokenizer = BPETokenizer(
    stoi,
    itos,
    merges
)


# ============================================================
# TOP-K
# ============================================================

def apply_top_k(
    logits,
    k
):

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


    cutoff = values[-1]


    return torch.where(
        logits < cutoff,
        torch.full_like(
            logits,
            float("-inf")
        ),
        logits
    )


# ============================================================
# TOP-P
# ============================================================

def apply_top_p(
    logits,
    p
):

    if p >= 1.0:

        return logits


    sorted_logits, indices = torch.sort(
        logits,
        descending=True
    )


    probabilities = torch.softmax(
        sorted_logits,
        dim=-1
    )


    cumulative = torch.cumsum(
        probabilities,
        dim=-1
    )


    remove = cumulative > p

    remove[0] = False


    sorted_logits = sorted_logits.masked_fill(
        remove,
        float("-inf")
    )


    filtered = torch.full_like(
        logits,
        float("-inf")
    )


    filtered.scatter_(
        0,
        indices,
        sorted_logits
    )


    return filtered


# ============================================================
# GENERATION
# ============================================================

def generate_answer(question):

    prompt = (
        "User: "
        + question.strip()
        + "\nEsakkiAI:"
    )


    prompt_tokens = tokenizer.encode(
        prompt
    )


    generated = torch.tensor(
        [prompt_tokens],
        dtype=torch.long,
        device=DEVICE
    )


    initial_length = generated.size(1)


    for _ in range(
        MAX_NEW_TOKENS
    ):

        context = generated[
            :,
            -checkpoint["max_seq_len"]:
        ]


        with torch.no_grad():

            logits = model(
                context
            )


        logits = logits[
            0,
            -1
        ]


        # ----------------------------------------------------
        # REPETITION PENALTY
        # ----------------------------------------------------

        previous_tokens = set(
            generated[0].tolist()
        )


        for token_id in previous_tokens:

            if logits[token_id] > 0:

                logits[token_id] /= (
                    REPETITION_PENALTY
                )

            else:

                logits[token_id] *= (
                    REPETITION_PENALTY
                )


        # ----------------------------------------------------
        # TEMPERATURE
        # ----------------------------------------------------

        logits = logits / TEMPERATURE


        # ----------------------------------------------------
        # TOP-K / TOP-P
        # ----------------------------------------------------

        logits = apply_top_k(
            logits,
            TOP_K
        )


        logits = apply_top_p(
            logits,
            TOP_P
        )


        probabilities = F.softmax(
            logits,
            dim=-1
        )


        next_token = torch.multinomial(
            probabilities,
            1
        ).item()


        generated = torch.cat(
            [
                generated,
                torch.tensor(
                    [[next_token]],
                    dtype=torch.long,
                    device=DEVICE
                )
            ],
            dim=1
        )


        # ----------------------------------------------------
        # CHECK GENERATED TEXT
        # ----------------------------------------------------

        new_text = tokenizer.decode(
            generated[
                0,
                initial_length:
            ].tolist()
        )


        # Stop before next conversation turn

        if "\nUser:" in new_text:

            new_text = new_text.split(
                "\nUser:",
                1
            )[0]

            break


        # Also stop if the model starts
        # another assistant turn

        if "\nEsakkiAI:" in new_text:

            new_text = new_text.split(
                "\nEsakkiAI:",
                1
            )[0]

            break


    return new_text.strip()


# ============================================================
# CHAT
# ============================================================

print()
print("=" * 60)

print("EsakkiAI Conversational Chat")

print("=" * 60)

print()
print("Temperature:", TEMPERATURE)
print("Top-K:", TOP_K)
print("Top-P:", TOP_P)
print("Max tokens:", MAX_NEW_TOKENS)

print()
print("Type 'exit' to stop.")
print()


while True:

    question = input(
        "User: "
    ).strip()


    if question.lower() == "exit":

        print(
            "EsakkiAI: Goodbye!"
        )

        break


    if not question:

        continue


    answer = generate_answer(
        question
    )


    print(
        "EsakkiAI:",
        answer
    )

    print()