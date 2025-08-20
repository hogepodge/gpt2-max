from max.graph import ops


def variance(x, axis=-1, correction=1):
    mean = ops.mean(x, axis=axis)
    return ops.mean((x - mean) ** 2, axis=axis)

def causal_mask(sequence_length, num_tokens, dtype, device):
    mask = ops.constant(float("-inf"), dtype=dtype, device=device)
    mask = ops.broadcast_to(
        mask, shape=(sequence_length, sequence_length + num_tokens)
    )
    return ops.band_part(mask, num_lower=None, num_upper=0, exclude=True)