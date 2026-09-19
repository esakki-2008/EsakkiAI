import os
import sys
import math

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

DATA_PATH = "data/train.txt"

DEVICE = torch.device("cpu")

SEQ_LEN = 64

EVALUATION_BATCHES = 100

MAX_NEW_TOKENS = 60

TEMPERATURE = 0.8

TOP_K = 40

TOP_P = 0.9

REPETITION_PENALTY = 1.15


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 60)
print("EsakkiAI Stage 21 Evaluation")
print("=" * 60)

print()
print("Loading dataset...")

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
# LOAD CHECKPOINT
# ============================================================

print()
print("Loading checkpoint...")

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


print(
    "Vocabulary:",
    vocab_size
)

print(
    "Model parameters:",
    sum(
        p.numel()
        for p in EsakkiGPT(
            vocab_size=vocab_size,
            d_model=checkpoint["d_model"],
            n_heads=checkpoint["n_heads"],
            n_layers=checkpoint["n_layers"],
            ff_dim=checkpoint["ff_dim"],
            max_seq_len=checkpoint["max_seq_len"]
        ).parameters()
    )
)

print(
    "Training checkpoint step:",
    checkpoint["step"]
)

print(
    "Best validation loss:",
    f'{checkpoint["best_val_loss"]:.4f}'
)


# ============================================================
# TOKENIZER
# ============================================================

class BPETokenizerForEvaluation:

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


tokenizer = BPETokenizerForEvaluation(
    stoi,
    itos,
    merges
)


# ============================================================
# ENCODE DATASET
# ============================================================

print()
print("Encoding dataset...")

encoded = tokenizer.encode(text)

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
    len(data) * 0.9
)

val_data = data[split_index:]


print(
    "Validation tokens:",
    len(val_data)
)


# ============================================================
# BUILD MODEL
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


# ============================================================
# VALIDATION LOSS
# ============================================================

print()
print("Calculating validation loss...")


def get_validation_batch():

    max_start = (
        len(val_data)
        - SEQ_LEN
        - 1
    )

    starts = torch.randint(
        0,
        max_start,
        (8,)
    )

    offsets = torch.arange(
        SEQ_LEN
    )

    x = val_data[
        starts[:, None] + offsets
    ]

    y = val_data[
        starts[:, None]
        + offsets
        + 1
    ]

    return (
        x.to(DEVICE),
        y.to(DEVICE)
    )


losses = []


with torch.no_grad():

    for _ in range(
        EVALUATION_BATCHES
    ):

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


validation_loss = (
    sum(losses)
    / len(losses)
)


# ============================================================
# PERPLEXITY
# ============================================================

perplexity = math.exp(
    validation_loss
)


print()
print(
    "Validation Loss:",
    f"{validation_loss:.4f}"
)

print(
    "Perplexity:",
    f"{perplexity:.2f}"
)


# ============================================================
# GENERATION
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
        -1
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


    cumulative = torch.cumsum(
        probabilities,
        dim=-1
    )


    remove = (
        cumulative > top_p
    )


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
        sorted_indices,
        sorted_logits
    )


    return filtered


def generate(prompt):

    input_tokens = tokenizer.encode(
        prompt
    )

    generated = torch.tensor(
        [input_tokens],
        dtype=torch.long,
        device=DEVICE
    )


    for _ in range(
        MAX_NEW_TOKENS
    ):

        context = generated[
            :,
            -SEQ_LEN:
        ]


        with torch.no_grad():

            logits = model(
                context
            )


        logits = logits[
            0,
            -1,
            :
        ]


        # Repetition penalty

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


        logits = logits / TEMPERATURE

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
                    device=DEVICE
                )
            ],
            dim=1
        )


    return tokenizer.decode(
        generated[0].tolist()
    )


# ============================================================
# TEST PROMPTS
# ============================================================

test_prompts = [

    "Artificial intelligence is",

    "Machine learning is",

    "A transformer model uses",

    "Python is a programming language that",

    "EsakkiAI is",

    "Neural networks are"

]


print()
print("=" * 60)
print("GENERATION TESTS")
print("=" * 60)


for prompt in test_prompts:

    print()
    print(
        "Prompt:",
        prompt
    )

    output = generate(
        prompt
    )


    if output.startswith(prompt):

        output = output[
            len(prompt):
        ]


    print(
        "Output:",
        output.strip()
    )


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 60)
print("ESAKKIAI EVALUATION REPORT")
print("=" * 60)

print()
print(
    f"Dataset characters : {len(text):,}"
)

print(
    f"Dataset tokens     : {len(data):,}"
)

print(
    f"Vocabulary size    : {vocab_size:,}"
)

print(
    f"Model parameters   : {sum(p.numel() for p in model.parameters()):,}"
)

print(
    f"Checkpoint step    : {checkpoint['step']:,}"
)

print(
    f"Validation loss    : {validation_loss:.4f}"
)

print(
    f"Perplexity         : {perplexity:.2f}"
)

print()
print(
    "Evaluation complete."
)

print(
    "Stage 21 complete."
)

print("=" * 60)