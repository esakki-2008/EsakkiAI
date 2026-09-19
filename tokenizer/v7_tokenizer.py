import re
from collections import Counter


class V7Tokenizer:

    SPECIAL_TOKENS = [
        "<PAD>",
        "<UNK>",
        "<USER>",
        "<ASSISTANT>",
        "<END>",
    ]

    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size

        self.stoi = {}
        self.itos = {}
        self.merges = []

    def split_text(self, text):
        """
        Split text while preserving V7 special tokens.
        """

        pattern = (
            r"<USER>|"
            r"<ASSISTANT>|"
            r"<END>|"
            r"[A-Za-z]+(?:'[A-Za-z]+)?|"
            r"[0-9]+|"
            r"[^\w\s]|"
            r"\s+"
        )

        return re.findall(pattern, text)

    def build_vocab(self, text):

        pieces = self.split_text(text)

        counter = Counter(pieces)

        # Start with special tokens.
        vocab = list(self.SPECIAL_TOKENS)

        # Add frequent pieces.
        for piece, _ in counter.most_common():

            if piece not in vocab:
                vocab.append(piece)

            if len(vocab) >= self.vocab_size:
                break

        self.stoi = {
            token: i
            for i, token in enumerate(vocab)
        }

        self.itos = {
            i: token
            for token, i in self.stoi.items()
        }

        return len(self.stoi)

    def encode(self, text):

        pieces = self.split_text(text)

        ids = []

        for piece in pieces:

            if piece in self.stoi:
                ids.append(
                    self.stoi[piece]
                )
            else:
                ids.append(
                    self.stoi["<UNK>"]
                )

        return ids

    def decode(self, ids):

        pieces = []

        for idx in ids:

            if idx in self.itos:

                pieces.append(
                    self.itos[idx]
                )

        return "".join(pieces)


def main():

    dataset_path = "data/conversation_v7.txt"

    with open(
        dataset_path,
        "r",
        encoding="utf-8"
    ) as f:

        text = f.read()

    tokenizer = V7Tokenizer(
        vocab_size=1000
    )

    vocab_size = tokenizer.build_vocab(
        text
    )

    print("=" * 60)
    print("EsakkiAI V7 Tokenizer Test")
    print("=" * 60)

    print(
        f"Vocabulary size: {vocab_size}"
    )

    test_text = (
        "<USER>\n"
        "What is machine learning?\n"
        "<ASSISTANT>\n"
        "Machine learning learns patterns from data.\n"
        "<END>\n"
    )

    encoded = tokenizer.encode(
        test_text
    )

    decoded = tokenizer.decode(
        encoded
    )

    print()
    print("Original:")
    print(test_text)

    print("Encoded:")
    print(encoded)

    print()
    print("Decoded:")
    print(decoded)

    print()
    print("Special token IDs:")

    for token in tokenizer.SPECIAL_TOKENS:

        print(
            f"{token}: "
            f"{tokenizer.stoi.get(token)}"
        )

    print()
    print("Special token test:")

    for token in tokenizer.SPECIAL_TOKENS:

        token_id = tokenizer.stoi[token]

        decoded_token = tokenizer.decode(
            [token_id]
        )

        print(
            f"{token} -> "
            f"{token_id} -> "
            f"{decoded_token}"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()