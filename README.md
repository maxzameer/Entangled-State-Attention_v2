# Entangled State Attention v2

Entangled State Attention (ESA) is a state-based causal sequence modeling layer that replaces explicit token-to-token attention score matrices with a causal state recurrence:

```text
E_t = A_t * E_{t-1} + B_t
```

ESA v2 exposes three named backends and a `compass()` utility for choosing Thunder's scan parameter `c`.

## Install

Local development install:

```bash
pip install -e .
```

With optional Triton dependency:

```bash
pip install -e .[triton]
```

## Backend names

```text
thunder = Optimized ESA backend
flare   = Triton ESA backend
pulse   = Base ESA backend
```

### Thunder

Thunder is the default optimized backend. It supports the chunked scan parameter `c`.

```python
from esa import ESA

layer = ESA(n_embd=128)
```

This is equivalent to:

```python
layer = ESA(
    n_embd=128,
    n_head=4,
    backend="thunder",
    c=16,
    precision="fp16",
)
```

Manual `c`:

```python
layer = ESA(n_embd=128, backend="thunder", c=32)
```

### Flare

Flare is the experimental Triton backend. It does not expose `c`.

```python
layer = ESA(n_embd=128, backend="flare")
```

Flare requires CUDA tensors and Triton. Install the optional dependency with:

```bash
pip install entangled-state-attention[triton]
```

### Pulse

Pulse is the base/reference backend. It does not expose `c`.

```python
layer = ESA(n_embd=128, backend="pulse")
```

Pulse is useful for correctness checks, debugging, and comparison.

## Important `c` rule

The chunked scan parameter `c` is only supported by `backend="thunder"`.

```python
ESA(n_embd=128, backend="thunder", c=16)  # valid
ESA(n_embd=128, backend="thunder", c=32)  # valid

ESA(n_embd=128, backend="flare", c=16)    # invalid
ESA(n_embd=128, backend="pulse", c=16)    # invalid
```

The error message is:

```text
The chunked scan parameter c is only supported by backend="thunder". backend="flare" and backend="pulse" do not expose c.
```

## Precision

Default precision:

```text
precision="fp16"
```

Supported precision strings:

```text
fp16 = default training mode
bf16 = optional mixed precision mode
fp32 = debug/stability mode
fp64 = correctness/reference mode
fp8  = experimental mode
```

`precision="fp8"` is accepted as an experimental mode, but the current safe implementation uses FP16 scan accumulation. A true FP8 kernel can be added later.

## Compass: selecting c

`compass()` evaluates Thunder ESA across candidate `c` values and recommends a practical value for the user’s workload.

```python
from esa import ESA, compass


def evaluate_fn(*, backend: str, c: int | None, precision: str):
    # Replace this with your own short training/evaluation loop.
    # Return val_loss or ppl, and tok_per_sec.
    ...

result = compass(
    evaluate_fn=evaluate_fn,
    c_candidates=(8, 16, 32, 64),
    reference_backend="pulse",
    precision="fp16",
    quality_tolerance=0.02,
)

layer = ESA(
    n_embd=128,
    backend="thunder",
    c=result.recommended,
    precision="fp16",
)
```

`CompassResult` contains:

```python
result.recommended   # best practical c
result.best_quality  # c with lowest validation loss / PPL
result.fastest       # c with highest tokens/sec
result.rows          # detailed metric rows
result.summary()     # human-readable recommendation
```

## Direct backend imports

```python
from esa.backends import ThunderESA, FlareESA, PulseESA

thunder = ThunderESA(n_embd=128, c=16)
flare = FlareESA(n_embd=128)
pulse = PulseESA(n_embd=128)
```

## Minimal forward example

```python
import torch
from esa import ESA

x = torch.randn(2, 128, 128)
layer = ESA(n_embd=128, n_head=4)
y = layer(x)
print(y.shape)  # torch.Size([2, 128, 128])
```

## Research note

ESA v2 is research software. Thunder is the default optimized backend, Flare is experimental, and Pulse is the reference backend. Results can depend on model size, sequence length, precision, GPU, dataset, and training setup.

## Citation

If you use Entangled State Attention v2, please cite:

```bibtex
@misc{hussain2026entangledstateattentionv2,
  title     = {Entangled State Attention: Efficient Long-Context Causal Modeling via Associative State Scans},
  author    = {Hussain, Zameer and Hussain, Akhtar},
  year      = {2026},
  publisher = {Zenodo},
  version   = {2.0.0},
  doi       = {10.5281/zenodo.21218821},
  url       = {https://doi.org/10.5281/zenodo.21218821}
}
```

Zenodo citation:

```text
Hussain, Z., & Hussain, A. (2026). Entangled State Attention: Efficient Long-Context Causal Modeling via Associative State Scans (2.0.0). Zenodo. https://doi.org/10.5281/zenodo.21218821
```

## License

Apache License 2.0.

Copyright 2026 Zameer Hussain and Akhtar Hussain.
