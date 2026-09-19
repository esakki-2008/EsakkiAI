import os
import sys
import re

import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, send_from_directory

# ------------------------------------------------------------
# Project path
# ------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)

from model.transformer import EsakkiGPT
from tokenizer.v7_tokenizer import V7Tokenizer


# ------------------------------------------------------------
# Flask
# ------------------------------------------------------------

app = Flask(
    __name__,
    static_folder="static"
)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "esakkiai_v7_best.pt"
)

DEVICE = torch.device("cpu")

TEMPERATURE = 0.65
TOP_K = 30
TOP_P = 0.90
REPETITION_PENALTY = 1.10
MAX_NEW_TOKENS = 80


# ------------------------------------------------------------
# Load V7
# ------------------------------------------------------------

print("=" * 60)
print("Loading EsakkiAI V7 Web Server")
print("=" * 60)

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

print(
    f"Model parameters: {parameter_count:,}"
)

print(
    f"Vocabulary size: {vocab_size}"
)

print(
    f"Best validation loss: "
    f"{checkpoint['val_loss']:.4f}"
)

print("EsakkiAI V7 loaded.")
print("=" * 60)


# ------------------------------------------------------------
# Sampling
# ------------------------------------------------------------

def apply_repetition_penalty(
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


# ------------------------------------------------------------
# Generate response
# ------------------------------------------------------------

@torch.no_grad()
def generate_response(question):

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

    if end_token_id in response_ids:

        response_ids = response_ids[
            :response_ids.index(
                end_token_id
            )
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

    response = response.replace(
        "<END>",
        ""
    )

    return response.strip()


# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------

@app.route("/")
def home():

    return send_from_directory(
        app.static_folder,
        "index.html"
    )


@app.route("/api/chat", methods=["POST"])
def chat():

    data = request.get_json(
        silent=True
    )

    if not data:

        return jsonify({
            "error": "Invalid request."
        }), 400

    message = data.get(
        "message",
        ""
    )

    if not isinstance(
        message,
        str
    ):

        return jsonify({
            "error": "Message must be text."
        }), 400

    message = message.strip()

    if not message:

        return jsonify({
            "error": "Message cannot be empty."
        }), 400

    if len(message) > 1000:

        return jsonify({
            "error": "Message is too long."
        }), 400

    print(
        f"User: {message}"
    )

    response = generate_response(
        message
    )

    print(
        f"EsakkiAI: {response}"
    )

    return jsonify({
        "response": response
    })


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------

if __name__ == "__main__":

    print()
    print("EsakkiAI is starting...")
    print()
    print(
        "EsakkiAI server is running."
    )

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )