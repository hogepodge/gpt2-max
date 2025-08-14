import re
import safetensors
from dataclasses import dataclass
from typing import Any
from collections.abc import Sequence


@dataclass
class WeightDict:
    """A dataclass that is designed to hold a map safetensor weights
    from a named entry in the source to a named tensor in a MAX Graph.
    """

    state: Any
    remapping_rules: Sequence[tuple[str, str]] = ()

    def __getitem__(self, name):
        return self.state.get_tensor(self.remap_name(name))
    
    def __contains__(self, name):
        return self.remap_name(name) in self.state.keys()
    
    def remap_name(self, name: str):
        for rule, replacement in self.remapping_rules:
            name = re.sub(rule, replacement, name)
        return name


def load_safetensors(path, rewrite_rules, framework="numpy"):
    st_weights = safetensors.safe_open(path, framework="numpy")
    weight_dict = WeightDict(
        st_weights,
        rewrite_rules
    )
    return weight_dict

