import os
import sys

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
    "checkpoints/esakkiai_v4_bpe_best.pt"
)

DEVICE = torch.device("cpu")

MAX_NEW_TOKENS = 100

TEMPERATURE = 0.8

TOP_K = 40

TOP_P = 0.9

REPETITION_PENALTY = 1.15


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print()
print("Loading EsakkiAI...")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)


# ============================================================
# REBUILD TOKENIZER DATA
# ============================================================

stoi = checkpoint["stoi"]

itos = checkpoint["itos"]

merges = checkpoint["merges"]

vocab_size = checkpoint["vocab_size"]


# ============================================================
# REBUILD MODEL
# ============================================================

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

model = model.to(DEVICE)

model.eval()


print(
    "Model loaded successfully."
)

print(
    "Vocabulary:",
    vocab_size
)

print(
    "Best validation loss:",
    f'{checkpoint["best_val_loss"]:.4f}'
)


# ============================================================
# BPE TOKENIZER RECONSTRUCTION
# ============================================================

class BPETokenizerForInference:

    def __init__(
        self,
        stoi,
        itos,
        merges
    ):

        self.stoi = stoi

        self.itos = {
            int(k): v
            for k, v in itos.items()
        }

        self.merges = [
            tuple(pair)
            for pair in merges
        ]


    def split_text(self, text):

        import re

        return re.findall(
            r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\w\s]|\s+",
            text
        )


    def apply_merges(self, pieces):

        for merge in self.merges:

            left, right = merge

            new_pieces = []

            i = 0

            while i < len(pieces):

                if (
                    i < len(pieces) - 1
                    and pieces[i] == left
                    and pieces[i + 1] == right
                ):

                    new_pieces.append(
                        left + right
                    )

                    i += 2

                else:

                    new_pieces.append(
                        pieces[i]
                    )

                    i += 1

            pieces = new_pieces

        return pieces


    def encode_word(self, word):

        if word.isspace():

            return [" "]


        pieces = list(word)

        pieces = self.apply_merges(
            pieces
        )

        return pieces


    def encode(self, text):

        tokens = []

        for word in self.split_text(text):

            pieces = self.encode_word(
                word
            )

            for piece in pieces:

                if piece in self.stoi:

                    tokens.append(
                        self.stoi[piece]
                    )

                else:

                    # Character fallback
                    for char in piece:

                        if char in self.stoi:

                            tokens.append(
                                self.stoi[char]
                            )

        return tokens


    def decode(self, tokens):

        pieces = []

        for token in tokens:

            if token in self.itos:

                pieces.append(
                    self.itos[token]
                )

        return "".join(pieces)


tokenizer = BPETokenizerForInference(
    stoi,
    itos,
    merges
)


# ============================================================
# TOP-K FILTER
# ============================================================

def apply_top_k(
    logits,
    top_k
):

    if top_k <= 0:

        return logits


    top_k = min(
        top_k,
        logits.size(-1)
    )


    values, _ = torch.topk(
        logits,
        top_k
    )


    minimum = values[
        ...,
        -1,
        None
    ]


    logits = torch.where(
        logits < minimum,
        torch.full_like(
            logits,
            float("-inf")
        ),
        logits
    )


    return logits


# ============================================================
# TOP-P FILTER
# ============================================================

def apply_top_p(
    logits,
    top_p
):

    if top_p >= 1.0:

        return logits


    sorted_logits, sorted_indices = torch.sort(
        logits,
        descending=True
    )


    probabilities = torch.softmax(
        sorted_logits,
        dim=-1
    )


    cumulative_probabilities = torch.cumsum(
        probabilities,
        dim=-1
    )


    remove = (
        cumulative_probabilities
        > top_p
    )


    # Keep at least one token
    remove[..., 0] = False


    sorted_logits = sorted_logits.masked_fill(
        remove,
        float("-inf")
    )


    logits = torch.full_like(
        logits,
        float("-inf")
    )


    logits.scatter_(
        -1,
        sorted_indices,
        sorted_logits
    )


    return logits


# ============================================================
# GENERATION
# ============================================================

def generate(
    prompt,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    top_k=TOP_K,
    top_p=TOP_P,
    repetition_penalty=REPETITION_PENALTY
):

    input_tokens = tokenizer.encode(
        prompt
    )


    if not input_tokens:

        return ""


    generated = torch.tensor(
        [input_tokens],
        dtype=torch.long,
        device=DEVICE
    )


    for _ in range(
        max_new_tokens
    ):

        # Keep only model context
        if (
            generated.size(1)
            > model.max_seq_len
        ):

            context = generated[
                :,
                -model.max_seq_len:
            ]

        else:

            context = generated


        with torch.no_grad():

            logits = model(
                context
            )


        logits = logits[
            0,
            -1,
            :
        ]


        # ----------------------------------------------------
        # REPETITION PENALTY
        # ----------------------------------------------------

        if repetition_penalty > 1.0:

            previous_tokens = set(
                generated[
                    0
                ].tolist()
            )


            for token_id in previous_tokens:

                if logits[token_id] > 0:

                    logits[token_id] /= (
                        repetition_penalty
                    )

                else:

                    logits[token_id] *= (
                        repetition_penalty
                    )


        # ----------------------------------------------------
        # TEMPERATURE
        # ----------------------------------------------------

        if temperature <= 0:

            next_token = torch.argmax(
                logits
            ).item()

        else:

            logits = logits / temperature


            # ------------------------------------------------
            # TOP-K
            # ------------------------------------------------

            logits = apply_top_k(
                logits,
                top_k
            )


            # ------------------------------------------------
            # TOP-P
            # ------------------------------------------------

            logits = apply_top_p(
                logits,
                top_p
            )


            probabilities = F.softmax(
                logits,
                dim=-1
            )


            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            ).item()


        next_token_tensor = torch.tensor(
            [[next_token]],
            dtype=torch.long,
            device=DEVICE
        )


        generated = torch.cat(
            [
                generated,
                next_token_tensor
            ],
            dim=1
        )


    generated_tokens = (
        generated[0].tolist()
    )


    return tokenizer.decode(
        generated_tokens
    )


# ============================================================
# INTERACTIVE MODE
# ============================================================

print()
print("=" * 60)

print("EsakkiAI Generation Test")

print("=" * 60)

print()
print("Controls:")
print("  temperature = randomness")
print("  top-k       = candidate limit")
print("  top-p       = probability nucleus")
print("  repetition  = repetition control")
print()
print("Type 'exit' to stop.")
print()


while True:

    prompt = input(
        "You: "
    ).strip()


    if prompt.lower() == "exit":

        print(
            "EsakkiAI: Goodbye!"
        )

        break


    if not prompt:

        continue


    print()
    print("EsakkiAI:", end=" ")


    result = generate(
        prompt
    )


    # Remove prompt from displayed output
    if result.startswith(prompt):

        result = result[
            len(prompt):
        ]


    print(
        result.strip()
    )

    print()