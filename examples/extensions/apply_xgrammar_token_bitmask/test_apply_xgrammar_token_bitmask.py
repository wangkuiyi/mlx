import mlx.core as mx
from mlx_sample_extensions import _ext

# Create example inputs
# Binary: 0b0101 (1st and 3rd positions allowed)
bitmask = mx.array([5], dtype=mx.uint32)  # Single int32 with bits 0 and 2 set (0b0101)
logits = mx.array([1.0, 2.0, 3.0, 4.0], dtype=mx.float32)

# Apply the bitmask
result = _ext.apply_xgrammar_token_bitmask(bitmask, logits)

print(result)

bitmask = mx.array([0xFFFFFFFF, 0x0], dtype=mx.uint32)  # All bits set
logits = mx.zeros((64,), dtype=mx.float32)

result = _ext.apply_xgrammar_token_bitmask(bitmask, logits)

print(result)
