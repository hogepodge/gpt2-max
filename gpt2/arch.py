from max.graph.weights import WeightsFormat
from max.nn.kv_cache import KVCacheStrategy
from max.interfaces import PipelineTask
from max.pipelines.lib import (
    SupportedArchitecture,
    SupportedEncoding,
    TextTokenizer,
)

from . import weight_adapters
from .model import GPT2

gpt2_arch = SupportedArchitecture(
    name="GPT2LMHeadModel",
    example_repo_ids=[
        "openai-community/gpt2",  # Add example model repository IDs
    ],
    default_encoding=SupportedEncoding.q4_k,
    supported_encodings={
        # Best guess based on the model
        SupportedEncoding.float32: [KVCacheStrategy.PAGED],
        SupportedEncoding.bfloat16: [KVCacheStrategy.PAGED],
    },
    pipeline_model=GPT2,
    tokenizer=TextTokenizer,
    # Probably need to implement weight importer
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=False,  # Start with single GPU for this small model
    weight_adapters={
        # TODO: add other weight formats
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    task=PipelineTask.TEXT_GENERATION,
)
