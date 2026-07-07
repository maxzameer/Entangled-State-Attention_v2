# Entangled State Attention v2

Entangled State Attention (ESA) is a state-based causal sequence modeling layer for causal sequence modeling.

Instead of building explicit token-to-token attention score matrices, ESA updates a causal state through a recurrence:

```text
E_t = A_t * E_{t-1} + B_t
```

ESA v2 provides:

```text
ESA              = main PyTorch layer
thunder          = optimized default backend
flare            = experimental Triton backend
pulse            = base/reference backend
compass()        = utility for selecting Thunder scan settings
thunderBoost()   = optional compile/warmup utility for layers and models
```

## Install

Local development install:

```bash
pip install -e .
```

With optional Triton dependency:

```bash
pip install -e .[triton]
```

From PyPI:

```bash
pip install entangled-state-attention
```

With optional Triton dependency:

```bash
pip install entangled-state-attention[triton]
```

## Quick start

```python
import torch
from esa import ESA

layer = ESA(
    embd=128,
    head=4,
    batch=16,
    block=1024,
    backend="thunder",
)

x = torch.randn(16, 1024, 128)
y = layer(x)

print(y.shape)
```

Output:

```text
torch.Size([16, 1024, 128])
```

## ESA layer API

Recommended usage:

```python
from esa import ESA

layer = ESA(
    embd=128,
    head=4,
    batch=16,
    block=1024,
    backend="thunder",
)
```

Arguments:

```text
embd      = embedding dimension
head      = number of heads
batch     = expected batch size
block     = expected sequence length
backend   = "thunder", "flare", or "pulse"
precision = scan precision mode
compass   = Thunder scan setting
```

## Backend names

```text
thunder = optimized ESA backend
flare   = experimental Triton ESA backend
pulse   = base/reference ESA backend
```

## Thunder backend

Thunder is the default optimized backend.

```python
from esa import ESA

layer = ESA(
    embd=128,
    head=4,
    batch=16,
    block=1024,
    backend="thunder",
)
```

Thunder supports the `compass` scan setting.

```python
layer = ESA(
    embd=128,
    head=4,
    batch=16,
    block=1024,
    backend="thunder",
    compass=16,
)
```

Most users do not need to set `compass` manually.

## Flare backend

Flare is the experimental Triton backend.

```python
from esa import ESA

layer = ESA(
    embd=128,
    head=4,
    batch=16,
    block=1024,
    backend="flare",
)
```

Flare requires CUDA tensors and Triton.

Install with:

```bash
pip install entangled-state-attention[triton]
```

## Pulse backend

Pulse is the base/reference backend.

```python
from esa import ESA

layer = ESA(
    embd=128,
    head=4,
    batch=16,
    block=1024,
    backend="pulse",
)
```

Pulse is useful for correctness checks, debugging, and comparison.

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

# thunderBoost

`thunderBoost()` is an optional utility for compiling and warming up an ESA layer or a full PyTorch model.

It runs warmup once at the module/model level.

It does not call `optimizer.step()`, so it does not train or update weights.

```python
from esa import thunderBoost
```

## Why use thunderBoost?

Without `thunderBoost`, a compiled model may spend the first few training steps compiling and warming up.

With `thunderBoost`, compile and warmup happen before real training starts.

```text
thunderBoost:
    compile + warmup first
    no optimizer step
    no weight update

training:
    starts clean and fast
```

## Single ESA layer

```python
from esa import ESA, thunderBoost

layer = ESA(
    embd=128,
    head=4,
    batch=16,
    block=1024,
    backend="thunder",
)

layer = thunderBoost(layer)
```

With state information:

```python
layer, state = thunderBoost(
    layer,
    state=True,
)

print(state)
```

## Multi-layer ESA model

For a model whose forward pass expects hidden vectors:

```python
model = thunderBoost(model)
```

If automatic shape detection is not possible, pass a sample batch tensor:

```python
import torch
from esa import thunderBoost

x = torch.randn(16, 1024, 128)

model = thunderBoost(
    model,
    batch=x,
)
```

With state information:

```python
model, state = thunderBoost(
    model,
    batch=x,
    state=True,
)

print(state)
```

## Tiny language model / LLM usage

For a causal language model, define a batch function.

```python
def batch(split="train"):
    source = train_data if split == "train" else val_data

    ix = torch.randint(
        low=0,
        high=len(source) - BLOCK_SIZE - 1,
        size=(BATCH_SIZE,),
    )

    x = torch.stack([source[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([source[i + 1:i + BLOCK_SIZE + 1] for i in ix])

    return x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
```

Then boost the full model:

```python
model = thunderBoost(
    model,
    batch=batch,
)
```

With state information:

```python
model, state = thunderBoost(
    model,
    batch=batch,
    state=True,
)

print(state)
```

For a model whose forward pass is:

```python
logits, loss = model(x, y)
```

`thunderBoost()` automatically detects the scalar loss when the model returns:

```python
(logits, loss)
```

## Dictionary-style LLM batch

`batch` can also return a dictionary.

```python
def batch(split="train"):
    data = next(dataloader)

    return {
        "input_ids": data["input_ids"].to(DEVICE),
        "labels": data["labels"].to(DEVICE),
        "attention_mask": data["attention_mask"].to(DEVICE),
    }
```

Then:

```python
model = thunderBoost(
    model,
    batch=batch,
)
```

Internally, dictionary batches are passed as:

```python
output = model(**batch)
```

If the model output contains:

```python
{"loss": loss}
```

the loss is detected automatically.

## thunderBoost return behavior

By default:

```python
model = thunderBoost(
    model,
    batch=batch,
)
```

returns only the boosted model.

With `state=True`:

```python
model, state = thunderBoost(
    model,
    batch=batch,
    state=True,
)
```

returns:

```text
model, state
```

Example state dictionary:

```python
{
    "compiled": True,
    "steps": 2,
    "compile_mode": "reduce-overhead",
    "device": "cuda",
    "backward": True,
    "amp": True,
    "peak_mem_mb": 123.45,
    "last_loss": 2.345,
    "auto_move_disabled_count": 4,
}
```

## thunderBoost arguments

```python
thunderBoost(
    module,
    batch=None,
    state=False,
    loss_fn=None,
    steps=None,
    compile=True,
    compile_mode=None,
    backward=True,
    amp=True,
    dtype=torch.float16,
    device="auto",
)
```

### module

Any `torch.nn.Module`.

Examples:

```python
layer = thunderBoost(layer)

model = thunderBoost(
    model,
    batch=batch,
)
```

### batch

`batch` can be:

```text
None
Tensor
tuple
dict
callable batch function
```

Examples:

```python
layer = thunderBoost(layer)

model = thunderBoost(
    model,
    batch=x,
)

model = thunderBoost(
    model,
    batch=(x, y),
)

model = thunderBoost(
    model,
    batch={
        "input_ids": input_ids,
        "labels": labels,
    },
)

model = thunderBoost(
    model,
    batch=batch,
)
```

### state

Default:

```python
state=False
```

So this returns only the model:

```python
model = thunderBoost(
    model,
    batch=batch,
)
```

This returns the model and state:

```python
model, state = thunderBoost(
    model,
    batch=batch,
    state=True,
)
```

### loss_fn

Optional function for extracting scalar loss from custom model outputs.

Example:

```python
model = thunderBoost(
    model,
    batch=batch,
    loss_fn=lambda output: output.my_loss,
)
```

### steps

Warmup steps.

If omitted, the default comes from:

```python
DEFAULT_BENCHMARK_CONFIG.compile_warmup_steps
```

### compile

Default:

```python
compile=True
```

Uses:

```python
torch.compile(module, mode=compile_mode, fullgraph=False)
```

### backward

Default:

```python
backward=True
```

Runs backward during warmup, but does not call `optimizer.step()`.

This warms the training path without changing weights.

### amp

Default:

```python
amp=True
```

Uses CUDA autocast when CUDA is available.

## Important device rule after thunderBoost

After `thunderBoost()`, inputs should already be on the same device as the model.

Correct:

```python
device = next(model.parameters()).device
x = torch.randn(16, 1024, 128, device=device)

y = model(x)
```

Avoid this after boosting:

```python
x = torch.randn(16, 1024, 128)
y = model(x)
```

During boosting, ESA internal auto device-copy is disabled to avoid CUDA graph partitioning from device-copy operations.

Normal non-boosted ESA can still auto-move inputs.

# Benchmark presets

ESA provides benchmark preset configs.

```python
from esa import (
    DEFAULT_BENCHMARK_CONFIG,
    FAST_BENCHMARK_CONFIG,
    PAPER_BENCHMARK_CONFIG,
)
```

## Default config

```python
from esa import DEFAULT_BENCHMARK_CONFIG

print(DEFAULT_BENCHMARK_CONFIG.compile_warmup_steps)
print(DEFAULT_BENCHMARK_CONFIG.speed_warmup_steps)
print(DEFAULT_BENCHMARK_CONFIG.speed_bench_steps)
print(DEFAULT_BENCHMARK_CONFIG.compile_mode)
```

## Dict-style benchmark defaults

```python
from esa import BENCHMARK_DEFAULTS

COMPILE_MODE = BENCHMARK_DEFAULTS["compile_mode"]
COMPILE_WARMUP_STEPS = BENCHMARK_DEFAULTS["compile_warmup_steps"]
SPEED_WARMUP_STEPS = BENCHMARK_DEFAULTS["speed_warmup_steps"]
SPEED_BENCH_STEPS = BENCHMARK_DEFAULTS["speed_bench_steps"]
```

Available dict presets:

```python
from esa import (
    BENCHMARK_DEFAULTS,
    FAST_BENCHMARK_DEFAULTS,
    PAPER_BENCHMARK_DEFAULTS,
)
```

Preset meanings:

```text
DEFAULT_BENCHMARK_CONFIG = balanced development benchmark
FAST_BENCHMARK_CONFIG    = quick smoke test
PAPER_BENCHMARK_CONFIG   = more stable paper/README benchmark
```

# Compass

`compass()` evaluates Thunder ESA across candidate scan settings and recommends a practical value for the user’s workload.

```python
from esa import ESA, compass


def evaluate_fn(*, backend: str, compass: int | None, precision: str):
    # Replace this with your own short training/evaluation loop.
    # Return validation loss or perplexity, and tokens/sec.
    ...


result = compass(
    evaluate_fn=evaluate_fn,
    compass_candidates=(8, 16, 32, 64),
    reference_backend="pulse",
    precision="fp16",
    quality_tolerance=0.02,
)

layer = ESA(
    embd=128,
    head=4,
    batch=16,
    block=1024,
    backend="thunder",
    compass=result.recommended,
    precision="fp16",
)
```

`CompassResult` contains:

```python
result.recommended   # best practical scan setting
result.best_quality  # setting with lowest validation loss / PPL
result.fastest       # setting with highest tokens/sec
result.rows          # detailed metric rows
result.summary()     # human-readable recommendation
```

# Minimal forward example

```python
import torch
from esa import ESA

x = torch.randn(2, 128, 128)

layer = ESA(
    embd=128,
    head=4,
    backend="thunder",
)

y = layer(x)

print(y.shape)
```

Output:

```text
torch.Size([2, 128, 128])
```

# TinyLM example

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from esa import ESA, thunderBoost


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 16
BLOCK_SIZE = 512
N_EMBD = 128
N_HEAD = 4
N_LAYER = 2


class ESABlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.ln1 = nn.LayerNorm(N_EMBD)

        self.esa = ESA(
            embd=N_EMBD,
            head=N_HEAD,
            batch=BATCH_SIZE,
            block=BLOCK_SIZE,
            backend="thunder",
        )

        self.ln2 = nn.LayerNorm(N_EMBD)

        self.ffn = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD, bias=False),
            nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD, bias=False),
        )

    def forward(self, x):
        x = x + self.esa(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class ESATinyLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, N_EMBD)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, N_EMBD)

        self.blocks = nn.ModuleList([
            ESABlock()
            for _ in range(N_LAYER)
        ])

        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        pos = torch.arange(0, T, device=idx.device)

        x = self.token_emb(idx) + self.pos_emb(pos)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(B * T, logits.size(-1)),
                targets.view(B * T),
            )

        return logits, loss


model = ESATinyLM(vocab_size).to(DEVICE)

model = thunderBoost(
    model,
    batch=batch,
)
```

# Research note

ESA v2 is research software.

Thunder is the default optimized backend, Flare is experimental, and Pulse is the reference backend.

Results can depend on model size, sequence length, precision, GPU, dataset, and training setup.

ESA does not automatically run benchmark steps during normal layer usage. Benchmark and warmup utilities are provided separately through `thunderBoost()` and benchmark config presets.

# Citation

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

# License

Apache License 2.0.

Copyright 2026 Zameer Hussain and Akhtar Hussain.
