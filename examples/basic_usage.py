from __future__ import annotations

import torch
from esa import ESA


x = torch.randn(2, 128, 64)

# Default: backend="thunder", compass=16, precision="fp16" on CUDA.
layer = ESA(embd=64, head=4)
y = layer(x)
print(y.shape)

# Manual Thunder compass.
layer_c32 = ESA(embd=64, head=4, backend="thunder", compass=32)
y = layer_c32(x)
print(y.shape)

# Pulse reference backend. No compass.
pulse = ESA(embd=64, head=4, backend="pulse")
y = pulse(x)
print(y.shape)
