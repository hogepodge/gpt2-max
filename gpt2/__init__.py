from .arch import gpt2_arch

# MAX looks for this variable when loading custom architectures
ARCHITECTURES = [gpt2_arch]

__all__ = ["gpt2_arch", "ARCHITECTURES"]
