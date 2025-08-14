import unittest
from gpt2.weights import load_safetensors, WeightDict
from gpt2.linear import Linear
from max.dtype import DType
from max.driver import CPU, Accelerator, accelerator_count
from max.graph import Graph
from max.experimental.tensor import Tensor

class TestWeightLoading(unittest.TestCase):

    def test_mapping(self):
        mapping = [("my_embedding", "embedding"),
                   ("my_attention", "attention")]
        weight_dict = load_safetensors("tests/test.safetensors", mapping)
        print(weight_dict)
        print(weight_dict["my_embedding"])
        print(weight_dict["my_attention"])

    def test_linear(self):
        device = Accelerator() if accelerator_count() else CPU()
        mapping = [("weight", "embedding"),
                   ("my_attention", "attention")]
        weight_dict = load_safetensors("tests/test.safetensors", mapping)
        linear = Linear(2, 2, bias=False, dtype=DType.float32, device=device)
        with Graph("test_linear") as graph:
            linear.load_weight_dict(weight_dict)
            print(linear.weight)

if __name__ == '__main__':
    unittest.main()