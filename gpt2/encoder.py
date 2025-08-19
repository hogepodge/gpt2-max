import tiktoken
from max.experimental.tensor import Tensor
from max.dtype import DType


class GPT2Encoder:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def encode(self, context, device):
        encoded = self.tokenizer.encode(context)
        l = len(encoded)
        
        return Tensor.constant(encoded, DType.int64, device).reshape([1,l])
    
    def decode(self, logits: Tensor):
        logits = logits[:, -1, :]
        m = logits.argmax()
        v = m.item()
        return self.tokenizer.decode([v])