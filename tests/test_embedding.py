import unittest

import re
import sys
import safetensors
import asyncio
import tiktoken
import numpy
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

from gpt2.embedding import Embedding
from gpt2.encoder import GPT2Encoder
from gpt2.statedict import StateDict
from gpt2.module import Module

from tests.pytorch_weights import load_state_dict
from tests.pytorch_gpt2 import GPTModel
from torch.nn import Embedding as TorchEmbedding
from torch import tensor as torchtensor
import torch
from torch.nn import Module as TorchModule


class GPT2PartialEmbeddings(Module):
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
        return x


class TestEmbedding(unittest.TestCase):

    def run_model(self, model, st_filename, state_mapping, context, device):
        st_state = safetensors.safe_open(st_filename, framework="numpy")


        token_type = TensorType(
            DType.int64, ("batch", "seqlen"), device=DeviceRef.from_device(device)
        )

        state = StateDict(st_state ,state_mapping)

        with Graph("TestGraph", input_types=(token_type,)) as graph:
            model.load_state_dict(state)
            graph.output(model(*graph.inputs))

        session = InferenceSession()
        
        # load the weights from the safetensor map
        weights = {name: state[name] for name, _ in model.named_parameters()}
        compiled = session.load(graph, weights_registry=weights)

        encoder = GPT2Encoder()
        input = encoder.encode(context, device)
        asyncio.run(input.realize)
        i = input.driver_tensor
        results = compiled(i)
        # TODO: check the actual results from the GPT2 model
        return results[0].to_numpy()

    def test_embedding(self):
        device = Accelerator() if accelerator_count() else CPU()
        st_filename = "gpt2-small-124M.safetensors"
        state_mapping = [
            ("weight", "tok_emb.weight"),
        ]

        model = Embedding(
            vocab_size=50257,
            dim=768,
            dtype=DType.float32,
            device=device,
        )

        self.run_model(model, st_filename, state_mapping, "The quick brown fox jumps over", device)

    def test_partial_model(self):
        device = Accelerator() if accelerator_count() else CPU()
        st_filename = "gpt2-small-124M.safetensors"
        state_mapping = [
            ("attention", "att"),
            ("feed_forward", "ff"),
            ("query", "W_query"),
            ("key", "W_key"),
            ("value", "W_value"),
            ("([^.]*_emb).weights", "\1.weight"),
        ]

        model = GPT2PartialEmbeddings(
            dim=768,
            num_layers=12,
            num_heads=12,
            vocab_size=50257,
            context_length=1024,
            qkv_bias=True,
            dtype=DType.float32,
            device=device,      
        )

        max_result = self.run_model(model, st_filename, state_mapping, "The quick brown fox jumps over", device)

        GPT_CONFIG_124M = {
            "vocab_size": 50257,     # Vocabulary size
            "context_length": 1024,  # Context length
            "emb_dim": 768,          # Embedding dimension
            "n_heads": 12,           # Number of attention heads
            "n_layers": 12,          # Number of layers
            "drop_rate": 0.0,        # Dropout rate
            "qkv_bias": True        # Query-Key-Value bias
        }

        torchmodel = GPTModel(GPT_CONFIG_124M)

        w = load_state_dict()
        torchmodel.load_weights(w)
        
        tokenizer = tiktoken.get_encoding("gpt2")
        encoded = tokenizer.encode("The quick brown fox jumps over")
        encoded_tensor = torchtensor(encoded).unsqueeze(0)
        print(encoded_tensor)

        torch_result = torchmodel(encoded_tensor)

        assert(numpy.allclose(torchmodel.embedding_checkpoint.detach().numpy(), max_result))

  

if __name__ == '__main__':
    unittest.main()