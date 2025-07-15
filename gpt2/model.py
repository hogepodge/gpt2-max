from typing import Dict, List, Optional
from max.pipelines.lib import PipelineModel, PipelineConfig, ModelInputs
from max.dtype import DType
from max.graph import Graph, TensorType, DeviceRef
from transformers import AutoConfig, AutoTokenizer
import numpy as np
from max.driver import Device, Tensor

from .model_config import GPT2Config
from .weight_adapters import convert_safetensor_state_dict

class GPT2Inputs(ModelInputs):
    """A class representing inputs for the GPT2 model.

    This class encapsulates the input tensors required for the Llama3 model
    execution.
    """

    tokens: Tensor
    """Tensor containing the input token IDs."""

    input_row_offsets: Tensor
    """Tensor containing the offsets for each row in the ragged input
    sequence."""

    signal_buffers: list[Tensor]
    """Device buffers used for synchronization in communication collectives."""

    return_n_logits: Tensor

    def __init__(
        self,
        tokens: Tensor,
        input_row_offsets: Tensor,
        signal_buffers: list[Tensor],
        return_n_logits: Tensor
    ) -> None:
        """
        Args:
            tokens: Input token IDs.
            input_row_offsets: Input row offsets (ragged tensors).
            signal_buffers: Device buffers used for synchronization in
                communication collectives.
        """
        self.tokens = tokens
        self.input_row_offsets = input_row_offsets
        self.signal_buffers = signal_buffers
        self.return_n_logits = return_n_logits



class GPT2(PipelineModel):
    """Main model class that implements your custom architecture."""

    def __init__(self, pipeline_config: GPT2Config, *args, **kwargs):
        super().__init__(pipeline_config=pipeline_config, *args, **kwargs)
        self.config = pipeline_config

    @classmethod
    def from_huggingface(cls, model_path: str, **kwargs) -> "GPT2":
        """Create MyModel instance from Hugging Face model."""
        # Load Hugging Face configuration
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        # Convert to our internal configuration format
        config = GPT2Config.from_huggingface_config(hf_config)

        return cls(config=config, **kwargs)
    
    @classmethod
    def calculate_max_seq_len(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        return 100
    
    def build_graph(self) -> Graph:
        """Build the computation graph for your model.

        This method defines how your model processes inputs and produces outputs.
        You'll implement the actual neural network logic here.
        """
        # Define input types for your model
        input_types = [
            TensorType(
                DType.int64,  # Token IDs
                shape=["batch_size", "sequence_length"],
                device=DeviceRef.GPU(),
            )
        ]

        # Create the computation graph
        with Graph("gpt2", input_types=input_types) as graph:
            # Get graph inputs
            (input_ids,) = graph.inputs

            # TODO: Implement your model's forward pass here
            # This is where you'd add your custom layers, attention mechanisms, etc.
            # For now, we'll add a placeholder

            # Example placeholder - replace with your actual model logic
            output = input_ids  # Placeholder

            # Set graph outputs
            graph.output(output)

        return graph

    def execute(self, model_inputs):
        pass
        # return super().execute(model_inputs)
    
    def prepare_initial_token_inputs(self, context_batch, kv_cache_inputs = None, return_n_logits = 1) -> GPT2Inputs:
        input_row_offsets = np.cumsum(
            [0] + [ctx.active_length for ctx in context_batch], dtype=np.uint32
        )

        # Create a ragged token vector of length: sum(len(t) for t in tokens).
        tokens = np.concatenate([ctx.next_tokens for ctx in context_batch])

        return GPT2Inputs(
            tokens=Tensor.from_numpy(tokens).to(self.devices[0]),
            input_row_offsets=Tensor.from_numpy(input_row_offsets).to(
                self.devices[0]
            ),
            signal_buffers=self.signal_buffers,
            return_n_logits=Tensor.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
        )
    
    def prepare_next_token_inputs(self, next_tokens, prev_model_inputs) -> GPT2Inputs:
        """Prepare the inputs for the next token in multistep execution.
        This should avoid any device synchronization or copy operations.
        """
        assert isinstance(prev_model_inputs, Llama3Inputs)
        row_offsets_size = prev_model_inputs.input_row_offsets.shape[0]
        next_row_offsets = self._input_row_offsets_prealloc[:row_offsets_size]

        return GPT2Inputs(
            tokens=next_tokens,
            input_row_offsets=next_row_offsets,
            signal_buffers=self.signal_buffers,
            return_n_logits=prev_model_inputs.return_n_logits,
        )
    
