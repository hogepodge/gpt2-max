from module import Module
from linear import Linear
from max.graph import ops


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