from dataclasses import dataclass
from typing import Any, Dict

from transformers import AutoConfig

@dataclass
class GPT2Config:
    """Configuration class for your custom model.

    This handles the translation between Hugging Face's config.json format
    and your model's internal parameter requirements for MAX graph building.
    """

    # Core model parameters
    vocab_size: int = 50257
    # hidden_size: int
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    max_sequence_length: int = 768

    # Add your model-specific parameters here
    # intermediate_size: int = 11008
    rms_norm_eps: float = 1e-5

    attn_pdrop: float = 0.1
    embd_pdrop: float = 0.1

    bos_token_id: int = 50256
    eos_token_id: int = 50256
    initializer_range: float = 0.02
    layer_norm_epsilon: float = 1e-05

    # "model_type": "gpt2",
    n_ctx: int = 1024
    n_positions: int = 1024
    resid_pdrop: float = 0.1
    #  summary_activation: null,
    summary_first_dropout: float = 0.1
    #  summary_proj_to_labels: true,
    #  summary_type: "cls_index",
    #  summary_use_proj: true,
    #  "task_specific_params": {
    #    "text-generation": {
    #      "do_sample": true,
    #      "max_length": 50
    #    }
    #  },

    @classmethod
    def from_huggingface_config(cls, hf_config: AutoConfig) -> "MyModelConfig":
        """Create MyModelConfig from Hugging Face AutoConfig."""
        return cls(
            vocab_size=hf_config.vocab_size,
#            hidden_size=hf_config.hidden_size,
            num_attention_heads=hf_config.n_head,
            num_hidden_layers=hf_config.n_layer,
            max_sequence_length=getattr(hf_config, "n_embd", 768),

            # Map other parameters from your Hugging Face config
            intermediate_size=getattr(hf_config, "intermediate_size", 11008),
            rms_norm_eps=getattr(hf_config, "layer_norm_epsilon", 1e-5),
        )
