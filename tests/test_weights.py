import unittest
from gpt2.weights import load_safetensors, WeightDict

class TestWeightLoading(unittest.TestCase):

    def test_mapping(self):
        mapping = [("my_embedding", "embedding"),
                   ("my_attention", "attention")]
        weight_dict = load_safetensors("tests/test.safetensors", mapping)
        print(weight_dict)
        print(weight_dict["my_embedding"])
        print(weight_dict["my_attention"])

if __name__ == '__main__':
    unittest.main()