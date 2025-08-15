from max.graph import ops, TensorValue, TensorType, Graph
from max.experimental.tensor import Tensor

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

