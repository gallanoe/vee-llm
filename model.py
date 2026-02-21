from __future__ import annotations
import numpy as np
import torch
import util
from torch import Tensor
from jaxtyping import Float, Int

from transformers import GPT2LMHeadModel, GPT2Tokenizer


class Tokenizer:
    def __init__(self):
        self.tok: GPT2Tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids)


def gelu(x):
    mid = np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)
    return 0.5 * x * (1 + torch.nn.functional.tanh(mid))


class LayerNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight: Float[Tensor, "dim"] = torch.nn.Parameter(torch.ones(dim))
        self.bias: Float[Tensor, "dim"] = torch.nn.Parameter(torch.zeros(dim))

    def forward(
        self, x: Float[Tensor, "batch_size seq_len dim"]
    ) -> Float[Tensor, "batch_size seq_len dim"]:
        mean = x.mean(dim=2, keepdim=True)
        variance = x.var(dim=2, correction=0, keepdim=True)
        x_hat = (x - mean) / torch.sqrt(variance + self.eps)
        return x_hat * self.weight + self.bias


class MultiLayerPerception(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.c_fc = torch.nn.Linear(dim, 4 * dim)
        self.c_proj = torch.nn.Linear(4 * dim, dim)

    def forward(
        self, x: Float[Tensor, "batch_size seq_len dim"]
    ) -> Float[Tensor, "batch_size seq_len dim"]:
        x = self.c_fc(x)
        x = gelu(x)
        return self.c_proj(x)


class CausalSelfAttention(torch.nn.Module):
    def __init__(self, n_heads: int, dim: int, max_seq_len: int):
        super().__init__()
        self.n_heads = n_heads
        assert dim % n_heads == 0
        self.head_dim = dim // n_heads
        self.c_attn = torch.nn.Linear(dim, 3 * dim)
        self.c_proj = torch.nn.Linear(dim, dim)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).bool(),
            persistent=False,
        )

    def forward(
        self, x: Float[Tensor, "batch_size seq_len dim"]
    ) -> Float[Tensor, "batch_size seq_len dim"]:
        batch_size, seq_len, dim = x.shape
        q: Float[Tensor, "batch_size seq_len dim"]
        k: Float[Tensor, "batch_size seq_len dim"]
        v: Float[Tensor, "batch_size seq_len dim"]
        q, k, v = self.c_attn(x).chunk(3, dim=2)
        qh: Float[Tensor, "batch_size n_heads seq_len head_dim"] = q.reshape(
            batch_size, seq_len, self.n_heads, self.head_dim
        ).transpose(1, 2)
        kh: Float[Tensor, "batch_size n_heads seq_len head_dim"] = k.reshape(
            batch_size, seq_len, self.n_heads, self.head_dim
        ).transpose(1, 2)
        vh: Float[Tensor, "batch_size n_heads seq_len head_dim"] = v.reshape(
            batch_size, seq_len, self.n_heads, self.head_dim
        ).transpose(1, 2)
        sh: Float[Tensor, "batch_size n_heads seq_len seq_len"] = (
            qh @ kh.transpose(2, 3) / np.sqrt(self.head_dim)
        ).masked_fill(
            ~self.mask[:seq_len, :seq_len], -torch.inf
        )  # pre-normalized scores
        sh = torch.nn.functional.softmax(sh, dim=3)
        wvh: Float[Tensor, "batch_size n_heads seq_len head_dim"] = sh @ vh
        return self.c_proj(wvh.transpose(1, 2).reshape(batch_size, seq_len, dim))


class TransformerBlock(torch.nn.Module):
    def __init__(self, n_heads: int, dim: int, max_seq_len: int):
        super().__init__()
        self.ln_1 = LayerNorm(dim)
        self.attn = CausalSelfAttention(n_heads, dim, max_seq_len)
        self.ln_2 = LayerNorm(dim)
        self.mlp = MultiLayerPerception(dim)

    def forward(
        self, x: Float[Tensor, "batch_size seq_len dim"]
    ) -> Float[Tensor, "batch_size seq_len dim"]:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        vocab_size = 50_257
        n_layers = 12
        n_heads = 12
        dim = 768
        max_seq_len = 1024

        self.wte = torch.nn.Embedding(vocab_size, dim)
        self.wpe = torch.nn.Embedding(max_seq_len, dim)
        self.h = torch.nn.ModuleList(
            [TransformerBlock(n_heads, dim, max_seq_len) for _ in range(n_layers)]
        )
        self.ln_f = LayerNorm(dim)

    def forward(
        self, x: Int[Tensor, "batch_size seq_len"]
    ) -> Float[Tensor, "batch_size seq_len dim"]:
        seq_len = x.shape[1]
        pos = torch.arange(seq_len, device=self.wte.weight.device)
        x = self.wte(x) + self.wpe(pos)
        for h in self.h:
            x = h(x)
        return self.ln_f(x)


class GPT2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = Transformer()

    def forward(
        self, x: Int[Tensor, "batch_size seq_len"]
    ) -> Float[Tensor, "batch_size seq_len vocab_size"]:
        x = self.transformer(x)
        return x @ self.transformer.wte.weight.T

    @classmethod
    def from_pretrained(cls, device: str):
        gpt = cls()
        ref: torch.nn.Module = GPT2LMHeadModel.from_pretrained("gpt2")  # type: ignore
        state_dict = util.transpose_state_dict(ref.state_dict())
        gpt.load_state_dict(state_dict)
        import gc

        del ref
        del state_dict
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
        gpt.to(device)
        return gpt
