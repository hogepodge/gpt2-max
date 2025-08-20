from module import Module
from linear import Linear
from functions import causal_mask
from max.graph.ops import softmax

class MultiHeadAttention(Module):
    def __init__(
        self, d_in, d_out, num_heads, *, qkv_bias=False, dtype, device
    ):
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        # Reduce the projection dim to match desired output dim
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.query = Linear(
            d_in, d_out, bias=qkv_bias, dtype=dtype, device=device
        )
        self.key = Linear(
            d_in, d_out, bias=qkv_bias, dtype=dtype, device=device
        )
        self.value = Linear(
            d_in, d_out, bias=qkv_bias, dtype=dtype, device=device
        )
        self.out_proj = Linear(d_out, d_out, dtype=dtype, device=device)
    
    def __call__(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.query(x)
        values = self.value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.reshape((b, num_tokens, self.num_heads, self.head_dim))
        values = values.reshape((b, num_tokens, self.num_heads, self.head_dim))
        queries = queries.reshape(
            (b, num_tokens, self.num_heads, self.head_dim)
        )

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)

        # Use the mask to fill attention scores
        # Mask is -inf for masked tokens, 0 for unmasked tokens
        mask = causal_mask(num_tokens, 0, dtype=x.dtype, device=x.device)
        attn_scores += mask

        attn_weights = softmax(attn_scores / self.head_dim**0.5)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.reshape((b, num_tokens, self.d_out))
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec
    
