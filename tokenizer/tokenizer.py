import re
from collections import Counter


class BPETokenizer:

    def __init__(self, vocab_size=1000):

        self.vocab_size = vocab_size

        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

        self.stoi = {}
        self.itos = {}

        self.merges = []

    # ==========================================================
    # BASIC TEXT SPLITTING
    # ==========================================================

    def split_text(self, text):

        return re.findall(
            r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\w\s]|\s+",
            text
        )

    # ==========================================================
    # INITIAL SYMBOL REPRESENTATION
    # ==========================================================

    def word_to_symbols(self, word):

        return list(word)

    # ==========================================================
    # COUNT SYMBOL PAIRS
    # ==========================================================

    def get_pair_counts(self, words):

        counts = Counter()

        for symbols, frequency in words.items():

            for i in range(len(symbols) - 1):

                pair = (
                    symbols[i],
                    symbols[i + 1]
                )

                counts[pair] += frequency

        return counts

    # ==========================================================
    # MERGE PAIR
    # ==========================================================

    def merge_pair(
        self,
        symbols,
        pair
    ):

        result = []

        i = 0

        while i < len(symbols):

            if (
                i < len(symbols) - 1
                and symbols[i] == pair[0]
                and symbols[i + 1] == pair[1]
            ):

                result.append(
                    symbols[i] + symbols[i + 1]
                )

                i += 2

            else:

                result.append(
                    symbols[i]
                )

                i += 1

        return result

    # ==========================================================
    # TRAIN BPE
    # ==========================================================

    def train(self, text):

        pieces = self.split_text(text)

        # Only words, numbers and punctuation participate
        # in BPE learning. Whitespace is preserved separately.

        words = [
            piece
            for piece in pieces
            if not piece.isspace()
        ]

        word_counts = Counter(words)

        symbol_words = {
            tuple(
                self.word_to_symbols(word)
            ): frequency
            for word, frequency in word_counts.items()
        }

        # Initial vocabulary

        vocabulary = set()

        for symbols in symbol_words:

            vocabulary.update(
                symbols
            )

        special_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token
        ]

        # Preserve whitespace as a token.

        vocabulary.add(" ")

        target_merges = max(
            0,
            self.vocab_size
            - len(vocabulary)
            - len(special_tokens)
        )

        # Learn merges

        for _ in range(target_merges):

            pair_counts = self.get_pair_counts(
                symbol_words
            )

            if not pair_counts:

                break

            best_pair, frequency = (
                pair_counts.most_common(1)[0]
            )

            if frequency < 2:

                break

            self.merges.append(
                best_pair
            )

            updated_words = {}

            for symbols, word_frequency in symbol_words.items():

                merged = self.merge_pair(
                    list(symbols),
                    best_pair
                )

                updated_words[
                    tuple(merged)
                ] = word_frequency

            symbol_words = updated_words

        # Build final vocabulary

        vocabulary = set()

        for symbols in symbol_words:

            vocabulary.update(
                symbols
            )

        # Add whitespace token

        vocabulary.add(" ")

        self.tokens = (
            special_tokens
            + sorted(vocabulary)
        )

        self.stoi = {
            token: index
            for index, token in enumerate(
                self.tokens
            )
        }

        self.itos = {
            index: token
            for token, index in self.stoi.items()
        }

    # ==========================================================
    # APPLY MERGES
    # ==========================================================

    def encode_word(self, word):

        # Preserve whitespace

        if word.isspace():

            return [" "]


        symbols = self.word_to_symbols(
            word
        )


        for pair in self.merges:

            symbols = self.merge_pair(
                symbols,
                pair
            )


        return symbols

    # ==========================================================
    # ENCODE TEXT
    # ==========================================================

    def encode(self, text):

        pieces = self.split_text(
            text
        )

        token_ids = []

        for piece in pieces:

            sub_tokens = self.encode_word(
                piece
            )

            for token in sub_tokens:

                token_id = self.stoi.get(
                    token,
                    self.stoi[self.unk_token]
                )

                token_ids.append(
                    token_id
                )

        return token_ids

    # ==========================================================
    # DECODE
    # ==========================================================

    def decode(self, token_ids):

        pieces = []

        for token_id in token_ids:

            token = self.itos.get(
                int(token_id),
                self.unk_token
            )

            if token in {
                self.pad_token,
                self.bos_token,
                self.eos_token,
                self.unk_token
            }:

                continue

            pieces.append(token)

        return "".join(
            pieces
        )

    # ==========================================================
    # VOCABULARY SIZE
    # ==========================================================

    @property
    def vocabulary_size(self):

        return len(self.tokens)


# ==============================================================
# TEST
# ==============================================================

if __name__ == "__main__":

    sample_text = """
    Artificial intelligence is a field of computer science.
    Machine learning allows computers to learn patterns from data.
    Transformers use attention mechanisms to process sequences.
    """

    tokenizer = BPETokenizer(
        vocab_size=200
    )

    tokenizer.train(
        sample_text
    )

    print()
    print("BPE tokenizer test successful!")
    print()

    print(
        "Vocabulary size:",
        tokenizer.vocabulary_size
    )

    text = "Artificial intelligence"

    encoded = tokenizer.encode(
        text
    )

    decoded = tokenizer.decode(
        encoded
    )

    print()
    print("Original:")
    print(text)

    print()
    print("Encoded:")
    print(encoded)

    print()
    print("Decoded:")
    print(decoded)

    print()
    print("Learned merges:")

    print(
        tokenizer.merges[:20]
    )