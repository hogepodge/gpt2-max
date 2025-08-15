from gpt2.module import Module
from max.graph import TensorValueLike, ops
from max.experimental.tensor import Tensor

class Embedding(Module):
    weight: TensorValueLike

    def __init__(self, vocab_size, dim, *, dtype, device):
        """ The Embedding class works as a lookup table to
        match tokens to embedding vectors.
        
        vocab_size: The size of the token vocabulary or context window
        
        dim: The embedding dimension size
            
        Internal weights is a tensor of shape (vocab_size, dim).
        """
        self.weight = Tensor.zeros((vocab_size, dim), dtype, device)

    def __call__(self, x):
        # TODO: with axis=-1 the error was really weird here
        return ops.gather(self.weight, x, axis=0)
