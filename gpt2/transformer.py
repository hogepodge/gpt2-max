from module import Module
from layernorm import LayerNorm
from attention import MultiHeadAttention
from sequential import Sequential
from linear import Linear
from functools import partial
from max.graph.ops import gelu

class FeedForward(Module):
    def __init__(self, dim, *, dtype, device):
        self.layers = Sequential(
            Linear(dim, 4 * dim, dtype=dtype, device=device),
            partial(gelu, approximate="tanh"),
            Linear(4 * dim, dim, dtype=dtype, device=device),
        )

    def __call__(self, x):
        return self.layers(x)

class TransformerBlock(Module):
    attention: MultiHeadAttention
    feed_forward: FeedForward
    norm1: LayerNorm
    norm2: LayerNorm

    def __init__(
        self, dim, num_heads, *, qkv_bias: bool = False, dtype, device
    ):
        self.attention = MultiHeadAttention(
            dim, dim, num_heads, qkv_bias=qkv_bias, dtype=dtype, device=device
        )
        self.feed_forward = FeedForward(dim, dtype=dtype, device=device)
        self.norm1 = LayerNorm(dim, dtype=dtype, device=device)
        self.norm2 = LayerNorm(dim, dtype=dtype, device=device)

    def __call__(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x)  # Shape [batch_size, num_tokens, emb_size]
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = x + shortcut  # Add the original input back

        return x