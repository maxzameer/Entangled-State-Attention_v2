from __future__ import annotations

import torch
from esa import ESA


x = torch.randn(2, 128, 64)

# Default: backend="thunder", c=16, precision="fp16" on CUDA.
layer = ESA(n_embd=64, n_head=4)
y = layer(x)
print(y.shape)

# Manual Thunder c.
layer_c32 = ESA(n_embd=64, n_head=4, backend="thunder", c=32)
y = layer_c32(x)
print(y.shape)

# Pulse reference backend. No c.
pulse = ESA(n_embd=64, n_head=4, backend="pulse")
y = pulse(x)
print(y.shape)
