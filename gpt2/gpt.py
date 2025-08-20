import sys
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial, reduce
from typing import Any

from encoder import GPT2Encoder
from module import Module, ModuleList
from mlp import MaxMLP
from attention import MultiHeadAttention
from embedding import Embedding
from linear import Linear
from functions import variance, causal_mask
from layernorm import LayerNorm
from transformer import TransformerBlock
from sequential import Sequential
from statedict import StateDict

import safetensors
from max.driver import CPU, Accelerator, accelerator_count
from max.dtype import DType
from max.engine.api import InferenceSession
from max.experimental.tensor import Tensor
from max.graph import (
    DeviceRef,
    Graph,
    TensorType,
    ops,
)

import asyncio

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

        encoder = GPT2Encoder()

        context = "The quick brown fox jump over"

        for i in range(20):


            input = encoder.encode(context, device)
            # input = Tensor.zeros([1, 1], DType.int64, device)
            asyncio.run(input.realize)
            print(input)
            i = input.driver_tensor
            results = compiled(i)
            next = encoder.decode(Tensor(storage=results[0]))
            print(next)
            context = context + next


        #print(Tensor(storage=results[0]))
