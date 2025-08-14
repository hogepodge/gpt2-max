import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial, reduce
from typing import Any

import safetensors
from max.driver import CPU, Accelerator, accelerator_count
from max.dtype import DType
from max.engine.api import InferenceSession
from max.experimental.tensor import Tensor
from max.graph import (
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    TensorValueLike,
    ops,
)

import asyncio


class Module:
    def named_children(self):
        for name, attr in vars(self).items():
            if isinstance(attr, (Module, Tensor, TensorValue)):
                yield name, attr

    def load_state_dict(self, state, path=()):
        for name, value in self.named_children():
            if isinstance(value, Module):
                value.load_state_dict(state, (*path, name))
            else:
                qualname = ".".join((*path, name))
                assert qualname in state, f"{qualname} not found in state"
                weight = ops.constant_external(name=qualname, type=value.type)
                setattr(self, name, weight)

    def named_parameters(self):
        for name, attr in self.named_children():
            if isinstance(attr, (Tensor, TensorValue)):
                yield name, attr
            elif isinstance(attr, Module):
                for subname, parameter in attr.named_parameters():
                    yield f"{name}.{subname}", parameter


class ModuleList(list, Module):
    def named_children(self):
        for i, attr in enumerate(self):
            if isinstance(attr, (Module, Tensor, TensorValue)):
                yield (str(i), attr)


class Linear(Module):
    weight: TensorValueLike
    bias: TensorValueLike = 0

    def __init__(self, in_dim, out_dim, *, bias: bool = True, dtype, device):
        self.weight = Tensor.zeros([in_dim, out_dim], dtype, device)
        self.bias = Tensor.zeros([out_dim], dtype, device) if bias else 0

    @property
    def __call__(self):
        return self.weight.shape[0]

    @property
    def out_dim(self):
        return self.weight.shape[1]

    def __call__(self, x):
        return x @ self.weight + self.bias


class MaxMLP(Module):
    c_fc: Linear
    c_proj: Linear

    def __init__(self, dim, hidden_dim, *, dtype, device):
        self.c_fc = Linear(dim, hidden_dim, dtype=dtype, device=device)
        self.c_proj = Linear(hidden_dim, dim, dtype=dtype, device=device)

    def __call__(self, x):
        expanded = self.c_fc(x)
        gelu_output = ops.gelu(expanded, approximate="tanh")
        return self.c_proj(gelu_output)


def causal_mask(sequence_length, num_tokens, dtype, device):
    mask = ops.constant(float("-inf"), dtype=dtype, device=device)
    mask = ops.broadcast_to(
        mask, shape=(sequence_length, sequence_length + num_tokens)
    )
    return ops.band_part(mask, num_lower=None, num_upper=0, exclude=True)


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

        attn_weights = ops.softmax(attn_scores / self.head_dim**0.5)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.reshape((b, num_tokens, self.d_out))
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec


def variance(x, axis=-1, correction=1):
    mean = ops.mean(x, axis=axis)
    return ops.mean((x - mean) ** 2, axis=axis)


class LayerNorm(Module):
    def __init__(self, dim, *, eps=1e-5, dtype, device):
        self.eps = eps
        self.scale = Tensor.zeros([dim], dtype, device)
        self.shift = Tensor.zeros([dim], dtype, device)

    def __call__(self, x):
        # TODO: file issue for device_ref bug
        # return ops.layer_norm(
        #     x, beta=self.scale, gamma=self.shift, epsilon=self.eps
        # )
        normed = (x - ops.mean(x)) / ops.sqrt(
            variance(x, correction=0) + self.eps
        )
        # TODO: file issue for Tensor * TensorValue from another graph
        return self.scale * normed + self.shift


class Sequential(ModuleList):
    def __init__(self, *layers):
        list.__init__(self, layers)

    def __call__(self, x):
        return reduce(lambda x, f: f(x), self, x)


class FeedForward(Module):
    def __init__(self, dim, *, dtype, device):
        self.layers = Sequential(
            Linear(dim, 4 * dim, dtype=dtype, device=device),
            partial(ops.gelu, approximate="tanh"),
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


class Embedding(Module):
    weight: TensorValueLike

    def __init__(self, vocab_size, dim, *, dtype, device):
        self.weight = Tensor.zeros((vocab_size, dim), dtype, device)

    def __call__(self, x):
        # TODO: with axis=-1 the error was really weird here
        return ops.gather(self.weight, x, axis=0)


class GPTModel(Module):
    def __init__(
        self,
        vocab_size,
        dim,
        num_layers,
        num_heads,
        context_length,
        *,
        qkv_bias: bool = False,
        dtype,
        device,
    ):
        self.tok_emb = Embedding(vocab_size, dim, dtype=dtype, device=device)
        self.pos_emb = Embedding(
            context_length, dim, dtype=dtype, device=device
        )

        self.trf_blocks = Sequential(
            *(
                TransformerBlock(
                    dim,
                    num_heads,
                    qkv_bias=qkv_bias,
                    dtype=dtype,
                    device=device,
                )
                for _ in range(num_layers)
            )
        )

        self.final_norm = LayerNorm(dim, dtype=dtype, device=device)
        self.out_head = Linear(
            dim, vocab_size, bias=False, dtype=dtype, device=device
        )

    def __call__(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            ops.range(
                0,
                seq_len,
                1,
                out_dim=seq_len,
                dtype=in_idx.dtype,
                device=in_idx.device,
            )
        )
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


@dataclass
class StateDict:
    state: Any
    name_remapping_rules: Sequence[tuple[str, str]] = ()

    def __getitem__(self, name):
        return self.state.get_tensor(self.remap_name(name))

    def __contains__(self, name):
        return self.remap_name(name) in self.state.keys()

    def remap_name(self, name: str):
        for rule, replacement in self.name_remapping_rules:
            name = re.sub(rule, replacement, name)
        return name


if __name__ == "__main__":
    path = sys.argv[1]
    st_state = safetensors.safe_open(path, framework="numpy")
    state = StateDict(
        st_state,
        [
            ("attention", "att"),
            ("feed_forward", "ff"),
            ("query", "W_query"),
            ("key", "W_key"),
            ("value", "W_value"),
            ("([^.]*_emb).weights", "\1.weight"),
        ],
    )

    device = Accelerator() if accelerator_count() else CPU()

    model = GPTModel(
        dim=768,
        num_layers=12,
        num_heads=12,
        vocab_size=50257,
        context_length=1024,
        qkv_bias=True,
        dtype=DType.float32,
        device=device,
    )

    token_type = TensorType(
        DType.int64, ("batch", "seqlen"), device=DeviceRef.from_device(device)
    )
    with Graph("GPT2", input_types=(token_type,)) as graph:
        model.load_state_dict(state)
        graph.output(model(*graph.inputs))

    session = InferenceSession()
    weights = {name: state[name] for name, _ in model.named_parameters()}
    compiled = session.load(graph, weights_registry=weights)
    input = Tensor.zeros([1, 1], DType.int64, device)
    asyncio.run(input.realize)
    i = input.driver_tensor
    results = compiled(i)
    print(Tensor(storage=results[0]))
