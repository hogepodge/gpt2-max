from typing import Dict, Any
from max.graph.weights import WeightData


def convert_safetensor_state_dict(
    state_dict: Dict[str, WeightData],
) -> Dict[str, WeightData]:
    """Convert SafeTensors weights to the format expected by your model.

    Args:
        state_dict: Raw weights loaded from SafeTensors format

    Returns:
        Converted weights ready for your model implementation
    """
    converted_weights = {}

    # TODO: implement the weight importer after checking that basic model loads

    #for key, weight in state_dict.items():
    #    # Apply any necessary transformations to weight names or values
    #    # This is where you handle differences between Hugging Face naming
    #    # conventions and what your model expects

        # Example: Remove prefixes that your model doesn't expect
    #    clean_key = key.replace("model.", "")

        # Example: Transpose weights if needed for your architecture
    #    if "linear" in clean_key and len(weight.shape) == 2:
            # Your model might expect different weight orientations
    #        converted_weights[clean_key] = weight  # Apply transpose if needed
    #    else:
    #        converted_weights[clean_key] = weight

    return converted_weights
