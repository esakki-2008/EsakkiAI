import torch
import torch.nn as nn


class SelfAttention(nn.Module):

    def __init__(
        self,
        d_model,
        n_heads
    ):

        super().__init__()

        assert d_model % n_heads == 0

        self.n_heads = n_heads

        self.head_dim = (
            d_model // n_heads
        )

        self.qkv = nn.Linear(
            d_model,
            d_model * 3
        )

        self.out = nn.Linear(
            d_model,
            d_model
        )


    def forward(self, x):

        batch_size, seq_len, d_model = x.shape

        qkv = self.qkv(x)

        q, k, v = qkv.chunk(
            3,
            dim=-1
        )


        q = q.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.head_dim
        ).transpose(1, 2)


        k = k.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.head_dim
        ).transpose(1, 2)


        v = v.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.head_dim
        ).transpose(1, 2)


        scores = (
            q @ k.transpose(-2, -1)
        )


        scores = scores / (
            self.head_dim ** 0.5
        )


        mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device
            ),
            diagonal=1
        ).bool()


        scores = scores.masked_fill(
            mask,
            float("-inf")
        )


        attention = torch.softmax(
            scores,
            dim=-1
        )


        output = attention @ v


        output = output.transpose(
            1,
            2
        ).contiguous()


        output = output.view(
            batch_size,
            seq_len,
            d_model
        )


        return self.out(output)


class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model,
        n_heads,
        ff_dim
    ):

        super().__init__()


        self.norm1 = nn.LayerNorm(
            d_model
        )


        self.attention = SelfAttention(
            d_model,
            n_heads
        )


        self.norm2 = nn.LayerNorm(
            d_model
        )


        self.feed_forward = nn.Sequential(

            nn.Linear(
                d_model,
                ff_dim
            ),

            nn.GELU(),

            nn.Linear(
                ff_dim,
                d_model
            )
        )


    def forward(self, x):

        x = x + self.attention(
            self.norm1(x)
        )

        x = x + self.feed_forward(
            self.norm2(x)
        )

        return x


class EsakkiGPT(nn.Module):

    def __init__(
        self,
        vocab_size=1000,
        d_model=128,
        n_heads=4,
        n_layers=4,
        ff_dim=512,
        max_seq_len=128
    ):

        super().__init__()


        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model
        )


        self.position_embedding = nn.Embedding(
            max_seq_len,
            d_model
        )


        self.blocks = nn.ModuleList(

            [
                TransformerBlock(
                    d_model,
                    n_heads,
                    ff_dim
                )

                for _ in range(n_layers)
            ]
        )


        self.norm = nn.LayerNorm(
            d_model
        )


        self.output = nn.Linear(
            d_model,
            vocab_size
        )


        self.max_seq_len = max_seq_len


    def forward(self, input_ids):

        batch_size, seq_len = (
            input_ids.shape
        )


        if seq_len > self.max_seq_len:

            raise ValueError(
                f"Sequence length {seq_len} "
                f"exceeds maximum "
                f"{self.max_seq_len}"
            )


        positions = torch.arange(
            seq_len,
            device=input_ids.device
        )


        x = (
            self.token_embedding(
                input_ids
            )
            +
            self.position_embedding(
                positions
            )
        )


        for block in self.blocks:

            x = block(x)


        x = self.norm(x)


        return self.output(x)