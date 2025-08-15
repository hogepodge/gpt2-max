import os
import urllib.request
from safetensors.torch import load_file
from torch.nn import Parameter as TorchParameter

def load_state_dict(model_config="gpt2-small (124M)"):

    BASE_CONFIG = {
        "vocab_size": 50257,    # Vocabulary size
        "context_length": 1024, # Context length
        "drop_rate": 0.0,       # Dropout rate
        "qkv_bias": True        # Query-key-value bias
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }


    BASE_CONFIG.update(model_configs[model_config])

    URL_DIR = {
        "gpt2-small (124M)": "gpt2",         # works ok
        "gpt2-medium (355M)": "gpt2-medium", # this file seems to have issues via `generate`
        "gpt2-large (774M)": "gpt2-large",   # works ok
        "gpt2-xl (1558M)": "gpt2-xl"         # works ok
    }

    url = f"https://huggingface.co/openai-community/{URL_DIR[model_config]}/resolve/main/model.safetensors"
    output_file = f"model-{URL_DIR[model_config]}.safetensors"

    # Download file
    if not os.path.exists(output_file):
        urllib.request.urlretrieve(url, output_file)

    # Load file
    state_dict = load_file(output_file)
    return state_dict


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return TorchParameter(right.detach())