from module import Module
from max.experimental.tensor import Tensor
from max.graph import ops
from functions import variance

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