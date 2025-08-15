import unittest

import re
import sys
import safetensors
import asyncio
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


class TestEmbedding(unittest.TestCase):

    def test_embedding(self):
    
        path = sys.argv[1]
        st_state = safetensors.safe_open("gpt2-small-124M.safetensors", framework="numpy")
        state = StateDict(
            st_state,
            [
                ("weight", "tok_emb.weight"),
            ],
        )

        device = Accelerator() if accelerator_count() else CPU()

        model = Embedding(
            vocab_size=50257,
            dim=768,
            dtype=DType.float32,
            device=device,
        )

        token_type = TensorType(
            DType.int64, ("batch", "seqlen"), device=DeviceRef.from_device(device)
        )
        
        with Graph("Embeddings", input_types=(token_type,)) as graph:
            print(model.named_parameters())
            model.load_state_dict(state)
            graph.output(model(*graph.inputs))

        session = InferenceSession()
        
        weights = {name: state[name] for name, _ in model.named_parameters()}
        compiled = session.load(graph, weights_registry=weights)

        encoder = GPT2Encoder()
        input = encoder.encode("The quick brown fox jumps over", device)

        asyncio.run(input.realize)
        print(input)

        i = input.driver_tensor
        results = compiled(i)
        print(results)


if __name__ == '__main__':
    unittest.main()