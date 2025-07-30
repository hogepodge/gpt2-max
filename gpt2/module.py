from gpt2.weights import WeightDict
from max.graph import ops, TensorValue
from gpt2.tensor import Tensor

class Module:
    def named_children(self):
        """Lists all of the named children in the Module"""
        for name, attr in vars(self).items():
            if isinstance(attr, (Module, Tensor, TensorValue)):
                yield name, attr

    def load_weight_dict(self, weights, path=()):
        for name, value in self.named_children():
            if isinstance(value, Module):
                value.load_weight_dict(weights, (*path, name))
            else:
                qualname = ".".join((*path, name))
                assert qualname in weights, f"{qualname} not found in weights"
                weight = ops.constant_external(name=qualname, type=value.type)
                setattr(self, name, weight)

    def named_parameters(self):
        for name, attr in self.named_children():
            if isinstance(attr, (Tensor, TensorValue)):
                yield name, attr
            elif isinstance(attr, Module):
                for subname, parameter in attr.named_parameters():
                    yield f"{name}.{subname}", parameter

