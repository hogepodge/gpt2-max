from module import Module
from max.experimental.tensor import Tensor
from max.graph import ops, TensorValue, TensorValueLike


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