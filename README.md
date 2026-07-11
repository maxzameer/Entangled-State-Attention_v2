# Entangled State Attention

**Entangled State Attention (ESA)** is a causal state-based sequence modeling architecture designed as an efficient alternative to conventional token-to-token self-attention.

Instead of constructing a full causal attention matrix, ESA updates a recurrent hidden state through an affine recurrence:

[
E_t = A_t \odot E_{t-1} + B_t
]

ESA v2.1 provides fast training backends, recurrent generation, a complete causal language model API, model saving/loading, and exact training checkpoint restoration.

---

## ESA v2.1

ESA v2.1 includes:

* **Flare** — Triton-accelerated ESA backend.
* **Thunder** — optimized PyTorch ESA backend.
* **Pulse** — reference ESA backend.
* **ESA-Lightning** — recurrent autoregressive generation engine.
* **ESAModel** — complete causal language model.
* **ESAModelConfig** — reproducible model configuration.
* **Trainer** — checkpoint, resume, best-model, and training-state management.
* **Model save/load** — portable model directories with configuration and metadata.

> **ESA-Lightning is a generation engine, not a fourth training backend.**

The public training backends are:

```text
flare
thunder
pulse
```

Generation is performed through:

```python
model.generate(...)
```

---

# Installation

## Install the current ESA v2.1 development branch

Until v2.1 is published to PyPI, install the current branch directly from GitHub:

```bash
pip install "git+https://github.com/maxzameer/Entangled-State-Attention_v2.git@esa-v2.1-update"
```

---

## Install from PyPI

```bash
pip install entangled-state-attention
```

> The PyPI release may lag behind the current development branch until the next package release.

---

## Install with Triton support

Flare requires Triton support.

```bash
pip install "entangled-state-attention[triton]"
```

For local development:

```bash
git clone https://github.com/maxzameer/Entangled-State-Attention_v2.git
cd Entangled-State-Attention_v2

pip install -e ".[triton]"
```

---

## Windows

On native Windows, Triton may require the Windows Triton distribution:

```bash
pip install triton-windows
```

Verify the installation:

```bash
python -c "import triton; print(triton.__version__)"
```

Then verify the ESA backends:

```bash
python -c "from esa import ESA; print(ESA(embd=32, head=4, backend='flare').backend); print(ESA(embd=32, head=4, backend='thunder').backend); print(ESA(embd=32, head=4, backend='pulse').backend)"
```

Expected output:

```text
flare
thunder
pulse
```

---

# Quick Start

## Standalone ESA Layer

```python
import torch

from esa import ESA

device = "cuda"

layer = ESA(
    embd=128,
    head=4,
    backend="flare",
).to(device)

x = torch.randn(
    2,
    128,
    128,
    device=device,
)

y = layer(x)

print(y.shape)
```

Output:

```text
torch.Size([2, 128, 128])
```

---

# ESA Backends

ESA v2.1 exposes three public sequence-processing backends.

## Flare

Flare is the default high-performance ESA training backend.

```python
from esa import ESA

layer = ESA(
    embd=128,
    head=4,
    backend="flare",
)
```

Flare is designed for:

* CUDA GPUs
* Triton-capable environments
* high-throughput ESA training
* long sequence workloads

Flare requires Triton.

---

## Thunder

Thunder is the optimized PyTorch ESA backend.

```python
from esa import ESA

layer = ESA(
    embd=128,
    head=4,
    backend="thunder",
)
```

Thunder can also use the `compass` configuration:

```python
layer = ESA(
    embd=128,
    head=4,
    backend="thunder",
    compass=16,
)
```

Thunder is useful when:

* Triton is unavailable
* a PyTorch-native execution path is preferred
* scan configuration experiments are required

---

## Pulse

Pulse is the reference ESA backend.

```python
from esa import ESA

layer = ESA(
    embd=128,
    head=4,
    backend="pulse",
)
```

Pulse is useful for:

* correctness testing
* reference comparisons
* backend validation
* debugging

---

# ESA Layer API

A larger ESA layer can be created as follows:

```python
from esa import ESA

layer = ESA(
    embd=384,
    head=6,
    batch=16,
    block=512,
    backend="flare",
    precision="fp16",
    dropout=0.1,
)
```

Common arguments:

| Argument    | Description                       |
| ----------- | --------------------------------- |
| `embd`      | Embedding dimension               |
| `head`      | Number of ESA heads               |
| `batch`     | Expected batch size metadata      |
| `block`     | Expected sequence length metadata |
| `backend`   | `flare`, `thunder`, or `pulse`    |
| `precision` | ESA numerical precision mode      |
| `compass`   | Thunder scan configuration        |
| `dropout`   | Dropout probability               |
| `gate_min`  | Minimum recurrent gate value      |
| `gate_max`  | Maximum recurrent gate value      |
| `eps`       | Numerical normalization epsilon   |

Example:

```python
layer = ESA(
    embd=384,
    head=6,
    backend="flare",
    precision="fp16",
)
```

---

# Precision

The default model precision configuration is:

```python
precision="fp16"
```

Available precision modes may include:

```text
fp16
bf16
fp32
fp64
fp8
```

Actual numerical behavior and hardware support depend on:

* selected backend
* GPU architecture
* PyTorch version
* CUDA version
* Triton version

---

# Complete ESA Language Model

ESA v2.1 includes a complete causal language model API.

```python
from esa import ESAModel, ESAModelConfig
```

---

## ESAModelConfig

`ESAModelConfig` stores the architecture and model settings.

```python
from esa import ESAModelConfig

config = ESAModelConfig(
    vocab_size=50257,
    block=512,
    n_layer=6,
    head=6,
    embd=384,
    dropout=0.1,
    bias=True,
    backend="flare",
    precision="fp16",
)
```

The current configuration API includes:

```text
vocab_size
block
n_layer
head
embd
dropout
bias
backend
precision
compass
gate_min
gate_max
eps
tie_embeddings
format_version
```

---

## ESAModel

`ESAModel` is the actual trainable causal language model.

```python
from esa import ESAModel

model = ESAModel(config)
```

Conceptually:

```text
ESAModelConfig
      ↓
describes the architecture
      ↓
ESAModel
      ↓
trainable neural network
```

---

## Direct Model Construction

`ESAModel` can also create its configuration internally:

```python
from esa import ESAModel

model = ESAModel(
    vocab_size=50257,
    block=512,
    n_layer=6,
    head=6,
    embd=384,
    dropout=0.1,
    backend="flare",
    precision="fp16",
)
```

For reproducible experiments, the explicit `ESAModelConfig` form is recommended.

---

# Training an ESA Language Model

`ESAModel.forward()` accepts token IDs and optional training targets.

```python
logits, loss = model(
    input_ids,
    targets=targets,
)
```

Example:

```python
import torch

from esa import ESAModel, ESAModelConfig

device = "cuda"

config = ESAModelConfig(
    vocab_size=50257,
    block=128,
    n_layer=6,
    head=6,
    embd=384,
    dropout=0.1,
    backend="flare",
    precision="fp16",
)

model = ESAModel(
    config
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
)

input_ids = torch.randint(
    0,
    config.vocab_size,
    (
        16,
        config.block,
    ),
    device=device,
)

targets = torch.randint(
    0,
    config.vocab_size,
    (
        16,
        config.block,
    ),
    device=device,
)

model.train()

optimizer.zero_grad(
    set_to_none=True
)

with torch.autocast(
    device_type="cuda",
    dtype=torch.float16,
):
    logits, loss = model(
        input_ids,
        targets=targets,
    )

loss.backward()

optimizer.step()

print(
    "Logits:",
    logits.shape,
)

print(
    "Loss:",
    float(loss),
)
```

---

# ESA-Lightning Generation

ESA v2.1 includes **ESA-Lightning**, a recurrent autoregressive generation engine.

Traditional autoregressive generation can repeatedly process previous tokens.

ESA-Lightning instead uses the recurrent ESA state:

```text
Prompt tokens
      ↓
ESA prefill
      ↓
Final recurrent state
      ↓
Generate next token
      ↓
Update recurrent state
      ↓
Generate next token
      ↓
...
```

Generation is exposed directly through:

```python
model.generate(...)
```

There is no public:

```python
backend="lightning"
```

training backend.

ESA-Lightning is automatically used by the model generation path.

---

# Text Generation

Example:

```python
result = model.generate(
    prompt="Once upon a time",
    tokenizer=tokenizer,
    max_new_tokens=128,
    temperature=0.8,
    top_k=50,
    seed=42,
)

print(result)
```

---

## Generation with Statistics

Use:

```python
return_result=True
```

to receive the full generation result.

```python
result = model.generate(
    prompt="Once upon a time",
    tokenizer=tokenizer,
    max_new_tokens=128,
    temperature=0.8,
    top_k=50,
    seed=42,
    compile=False,
    return_result=True,
)

print(result.text)

print(
    "Prompt tokens:",
    result.stats.prompt_tokens,
)

print(
    "Generated tokens:",
    result.stats.generated_tokens,
)

print(
    "Decode steps:",
    result.stats.decode_steps,
)

print(
    "Decode tokens/sec:",
    result.stats.decode_tok_s,
)

print(
    "State memory MB:",
    result.stats.state_mb,
)
```

A generation result can expose:

```text
sequences
generated_ids
stats
text
```

Generation statistics may include:

```text
prompt_tokens
prefill_tokens
generated_tokens
decode_steps
prefill_seconds
decode_seconds
decode_tok_s
total_seconds
state_bytes
state_mb
```

---

# Tensor-Based Generation

Generation can also begin from token IDs.

```python
output = model.generate(
    input_ids=input_ids,
    max_new_tokens=128,
)
```

---

# Generation Parameters

The public generation API supports parameters including:

```text
input_ids
prompt
tokenizer
max_new_tokens
temperature
top_k
top_p
eos_token_id
seed
compile
compile_mode
progress_every
return_result
```

Example:

```python
result = model.generate(
    prompt="The future of efficient AI is",
    tokenizer=tokenizer,
    max_new_tokens=200,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    seed=42,
    compile=True,
    compile_mode="reduce-overhead",
    return_result=True,
)
```

---

# Low-Level ESA-Lightning API

ESA layers expose recurrent generation operations.

## Initialize State

```python
state = layer.init_state(
    batch=1,
)
```

---

## Prompt Prefill

```python
output, state = layer.prefill(
    x
)
```

The prompt is processed and converted into a recurrent ESA state.

---

## One-Token Decode

```python
output, state = layer.decode_step(
    x_next,
    state,
)
```

The state is updated without replaying the full previous token sequence through the ESA recurrence.

---

# Saving a Model

Save a complete ESA model:

```python
model.save(
    "my_esa_model"
)
```

With metadata:

```python
model.save(
    "my_esa_model",
    metadata={
        "dataset": "TinyStories",
        "step": 20000,
        "batch_size": 16,
        "block_size": 128,
    },
)
```

A saved model directory contains:

```text
my_esa_model/
├── config.json
├── metadata.json
└── model.pt
```

---

# Loading a Model

Load a saved model:

```python
from esa import ESAModel

model = ESAModel.load(
    "my_esa_model",
    device="cuda",
)
```

The configuration and model weights are restored automatically.

---

# Complete Save and Load Example

```python
from esa import ESAModel

model.save(
    "trained_esa_model",
    metadata={
        "dataset": "my_dataset",
        "step": 20000,
    },
)

loaded_model = ESAModel.load(
    "trained_esa_model",
    device="cuda",
)

loaded_model.eval()
```

---

# Trainer

ESA v2.1 includes a training checkpoint manager:

```python
from esa import Trainer
```

Example:

```python
trainer = Trainer(
    model,
    optimizer=optimizer,
    checkpoint_dir="checkpoints",
    save_every=1000,
    save_at=[
        5000,
        10000,
    ],
    save_best=True,
    save_last=True,
    keep_last_n=3,
)
```

The Trainer can manage:

* periodic checkpoints
* explicitly requested checkpoint steps
* best-validation checkpoints
* last checkpoints
* checkpoint pruning
* training-state restoration

---

# Save a Training Checkpoint

Set the current training step:

```python
trainer.state.step = 1000
```

Then save:

```python
checkpoint_path = trainer.save_checkpoint(
    step=1000,
)
```

A training checkpoint contains:

```text
checkpoint/
├── config.json
├── metadata.json
├── model.pt
└── training_state.pt
```

---

# Training State

A training checkpoint can restore:

```text
model weights
optimizer state
scheduler state
gradient scaler state
training step
best validation loss
Python random state
PyTorch random state
CUDA random state
extra metadata
```

This makes it possible to continue interrupted training without starting from scratch.

---

# Load a Checkpoint

```python
trainer.load_checkpoint(
    "checkpoints/step_001000",
    device="cuda",
)
```

The same Trainer instance is updated with the restored state.

---

# Resume Training

The Trainer exposes:

```python
trainer.resume_from(...)
```

Example:

```python
trainer.resume_from(
    "last",
    device="cuda",
)
```

Other supported checkpoint selectors may include:

```text
last
latest
best
```

An explicit checkpoint path can also be used.

```python
trainer.resume_from(
    "checkpoints/step_010000",
    device="cuda",
)
```

---

# Continue Training After Resume

```python
trainer.load_checkpoint(
    checkpoint_path,
    device="cuda",
)

model = trainer.model
optimizer = trainer.optimizer

model.train()

optimizer.zero_grad(
    set_to_none=True
)

logits, loss = model(
    input_ids,
    targets=targets,
)

loss.backward()

optimizer.step()

trainer.state.step += 1
```

---

# Exact Resume Verification

ESA v2.1 checkpoint restoration has been verified with:

```text
Model weight difference      : 0.0
Restored forward difference  : 0.0
Optimizer states             : fully restored
Training step                : restored
Continued training           : successful
```

---

# Automatic Checkpoint Management

During training:

```python
trainer.maybe_save(
    step=step,
    val_loss=val_loss,
)
```

This can be used to manage:

```text
periodic checkpoints
requested checkpoint steps
best validation checkpoint
last checkpoint
checkpoint pruning
```

---

# Save Final Training State

At the end of training:

```python
trainer.save_final()
```

---

# thunderBoost

ESA includes an optional `thunderBoost()` utility.

```python
from esa import thunderBoost
```

Example:

```python
model = thunderBoost(
    model,
    batch=batch,
)
```

It can be used for:

* `torch.compile`
* warmup forward passes
* warmup backward passes
* AMP warmup

The warmup utility does not intentionally perform an optimizer update.

State information can also be requested:

```python
model, state = thunderBoost(
    model,
    batch=batch,
    state=True,
)

print(state)
```

---

# Compass

ESA includes a `compass()` utility for Thunder configuration experiments.

```python
from esa import compass
```

Example:

```python
result = compass(
    evaluate_fn=evaluate_fn,
    compass_candidates=(
        8,
        16,
        32,
        64,
    ),
    reference_backend="pulse",
    precision="fp16",
    quality_tolerance=0.02,
)

print(
    result.recommended
)
```

A Compass result may expose:

```text
recommended
best_quality
fastest
rows
summary()
```

Compass is intended for Thunder backend configuration experiments.

---

# Recommended Backend Selection

## Flare

Recommended starting point for:

```text
GPU training
Triton-capable systems
high-throughput ESA workloads
```

Example:

```python
backend="flare"
```

---

## Thunder

Recommended when:

```text
a PyTorch-native optimized path is preferred
Triton is unavailable
Thunder scan configuration is being tested
```

Example:

```python
backend="thunder"
```

---

## Pulse

Recommended for:

```text
reference behavior
correctness testing
debugging
backend comparisons
```

Example:

```python
backend="pulse"
```

---

## ESA-Lightning

Used for:

```text
autoregressive text generation
recurrent token decoding
constant recurrent-state decoding
```

ESA-Lightning is accessed through:

```python
model.generate(...)
```

---

# Example: Small ESA Language Model

```python
import torch

from esa import (
    ESAModel,
    ESAModelConfig,
)

device = "cuda"

config = ESAModelConfig(
    vocab_size=50257,
    block=128,
    n_layer=6,
    head=6,
    embd=384,
    dropout=0.1,
    bias=True,
    backend="flare",
    precision="fp16",
)

model = ESAModel(
    config
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
)

input_ids = torch.randint(
    0,
    config.vocab_size,
    (
        16,
        config.block,
    ),
    device=device,
)

targets = torch.randint(
    0,
    config.vocab_size,
    (
        16,
        config.block,
    ),
    device=device,
)

model.train()

optimizer.zero_grad(
    set_to_none=True
)

with torch.autocast(
    device_type="cuda",
    dtype=torch.float16,
):
    logits, loss = model(
        input_ids,
        targets=targets,
    )

loss.backward()

optimizer.step()

print(
    "Logits shape:",
    logits.shape,
)

print(
    "Loss:",
    float(loss),
)
```

---

# Example: Generate Text from a Trained Model

```python
from esa import ESAModel

model = ESAModel.load(
    "trained_esa_model",
    device="cuda",
)

model.eval()

result = model.generate(
    prompt="Once upon a time",
    tokenizer=tokenizer,
    max_new_tokens=200,
    temperature=0.8,
    top_k=50,
    seed=42,
    compile=False,
    return_result=True,
)

print(result.text)

print(
    "Decode tokens/sec:",
    result.stats.decode_tok_s,
)

print(
    "State memory MB:",
    result.stats.state_mb,
)
```

---

# Example: Dual-GPU ESA vs Attention Experiment

A matched experiment can run:

```text
GPU 0 → ESA
GPU 1 → Attention
```

Example configuration:

```text
Batch size    : 16
Block size    : 128
ESA backend   : Flare
Attention     : PyTorch SDPA
```

This allows simultaneous comparison of:

```text
training loss
validation loss
perplexity
training throughput
wall-clock throughput
GPU memory
elapsed time
ETA
generated text
generation speed
```

---

# Kaggle

ESA v2.1 has been verified on Kaggle with:

```text
Python       : 3.12.13
PyTorch      : 2.10.0+cu128
CUDA         : 12.8
Triton       : 3.6.0
GPU          : Tesla T4
```

A fresh Kaggle notebook can install the development branch with:

```python
import sys
import subprocess

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "git+https://github.com/maxzameer/Entangled-State-Attention_v2.git@esa-v2.1-update",
    ],
    check=True,
)
```

Verify:

```python
import torch
import triton
import esa

from esa import (
    ESA,
    ESAModel,
    ESAModelConfig,
    Trainer,
)

print(
    "ESA:",
    esa.__file__,
)

print(
    "PyTorch:",
    torch.__version__,
)

print(
    "CUDA:",
    torch.version.cuda,
)

print(
    "Triton:",
    triton.__version__,
)

print(
    "GPU count:",
    torch.cuda.device_count(),
)
```

---

# Verified ESA v2.1 Tests

ESA v2.1 has been tested on:

```text
Windows local environment
Kaggle Tesla T4 environment
```

Backend smoke tests:

```text
Flare    PASS
Thunder  PASS
Pulse    PASS
```

Verified backend operations:

```text
FP16 forward
backward gradients
finite outputs
finite gradients
```

Verified model lifecycle:

```text
ESAModel creation
Flare training forward
backward propagation
optimizer step
model.generate()
ESA-Lightning generation
generation statistics
model.save()
ESAModel.load()
exact configuration restoration
exact weight restoration
loaded-model forward equivalence
```

Verified checkpoint lifecycle:

```text
checkpoint save
model state restoration
optimizer state restoration
training step restoration
identical restored forward output
continued training after resume
```

Package test suite:

```text
10 passed
```

---

# Verified Model Lifecycle Result

A complete ESA v2.1 lifecycle test produced:

```text
Model creation              PASS
Training forward            PASS
Backward gradients          PASS
Optimizer step              PASS
Generation                  PASS
Model save                  PASS
Model load                  PASS
Maximum weight difference   0.0
Maximum forward difference  0.0
```

---

# Verified Trainer Resume Result

A complete Trainer checkpoint test produced:

```text
Checkpoint saved            PASS
Restored step               1
Maximum model difference    0.0
Optimizer states restored   24 / 24
Maximum forward difference  0.0
Continued training          PASS
New training step           2
```

---

# ESA Architecture

The core ESA recurrence is:

[
E_t = A_t \odot E_{t-1} + B_t
]

where:

* (E_t) is the recurrent entangled state
* (A_t) controls state retention
* (B_t) writes new information into the state
* (\odot) is elementwise multiplication

This recurrence can be represented as an affine transform:

[
(A_t, B_t)
]

with associative composition:

[
(A_b, B_b) \circ (A_a, B_a)
===========================

(A_b \odot A_a,;
A_b \odot B_a + B_b)
]

This associative structure allows the recurrent computation to be reorganized into parallel scan implementations.

---

# Why ESA?

Conventional causal self-attention computes relationships between token pairs.

ESA instead maintains an evolving causal state.

Conceptually:

```text
Self-Attention

token
  ↓
Q, K, V
  ↓
token-to-token interaction
  ↓
attention matrix
  ↓
output
```

```text
ESA

token
  ↓
gate + state update
  ↓
causal recurrent state
  ↓
readout
  ↓
output
```

The goal is to explore efficient causal sequence modeling with:

* associative state scans
* recurrent decoding
* reduced state growth during generation
* alternative long-context computation patterns

---

# Training vs Generation

ESA v2.1 intentionally separates high-throughput training from recurrent generation.

## Training

Use:

```text
Flare
Thunder
Pulse
```

Example:

```python
config = ESAModelConfig(
    vocab_size=50257,
    backend="flare",
)
```

---

## Generation

Use:

```python
model.generate(...)
```

The model uses the ESA-Lightning recurrent generation path.

This separation allows:

```text
parallel training
+
recurrent token-by-token inference
```

---

# Dropout

The default `ESAModelConfig` dropout is:

```python
dropout=0.1
```

which corresponds to:

```text
10% dropout
```

Example:

```python
config = ESAModelConfig(
    vocab_size=50257,
    dropout=0.1,
)
```

For deterministic save/load equivalence tests, dropout can be disabled:

```python
dropout=0.0
```

---

# Reproducibility

For reproducible experiments:

```python
import torch

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

Generation also accepts an explicit seed:

```python
result = model.generate(
    prompt="Once upon a time",
    tokenizer=tokenizer,
    max_new_tokens=128,
    seed=42,
)
```

---

# Benchmarking Guidance

ESA performance depends on:

* GPU architecture
* CUDA version
* PyTorch version
* Triton version
* backend
* precision
* batch size
* sequence length
* embedding dimension
* number of layers
* model compilation
* warmup
* dataset
* optimizer configuration

Fair comparisons should report all relevant experimental settings.

For matched ESA vs Attention experiments, keep constant:

```text
dataset
tokenizer
vocabulary
batch size
block size
number of layers
embedding dimension
training steps
optimizer
learning-rate schedule
evaluation batches
random seeds
```

Parameter counts should also be reported.

---

# Research Status

ESA is research software.

The project explores alternative causal sequence-modeling architectures and efficient recurrent state computation.

Results may vary significantly across:

```text
hardware
sequence length
batch size
precision
backend
model size
dataset
compiler configuration
```

No single backend should be assumed to be universally fastest for every workload.

---

# Project Structure

A typical ESA repository structure is:

```text
Entangled-State-Attention_v2/
├── esa/
│   ├── __init__.py
│   ├── layer.py
│   ├── model.py
│   ├── generation.py
│   ├── trainer.py
│   └── backends/
│       ├── flare.py
│       ├── thunder.py
│       └── pulse.py
├── tests/
├── README.md
├── pyproject.toml
└── LICENSE
```

---

# Public API

Common imports:

```python
from esa import (
    ESA,
    ESAModel,
    ESAModelConfig,
    Trainer,
)
```

Depending on the installed release, additional utilities may include:

```python
from esa import (
    compass,
    thunderBoost,
)
```

---

# API Summary

## ESA

```python
layer = ESA(
    embd=128,
    head=4,
    backend="flare",
)
```

---

## ESAModelConfig

```python
config = ESAModelConfig(
    vocab_size=50257,
    block=512,
    n_layer=6,
    head=6,
    embd=384,
    backend="flare",
)
```

---

## ESAModel

```python
model = ESAModel(
    config
)
```

---

## Forward

```python
logits, loss = model(
    input_ids,
    targets=targets,
)
```

---

## Generate

```python
result = model.generate(
    prompt=prompt,
    tokenizer=tokenizer,
    max_new_tokens=128,
)
```

---

## Save

```python
model.save(
    "model_directory"
)
```

---

## Load

```python
model = ESAModel.load(
    "model_directory",
    device="cuda",
)
```

---

## Trainer

```python
trainer = Trainer(
    model,
    optimizer=optimizer,
)
```

---

## Save Checkpoint

```python
trainer.save_checkpoint(
    step=1000,
)
```

---

## Load Checkpoint

```python
trainer.load_checkpoint(
    checkpoint_path,
    device="cuda",
)
```

---

## Resume

```python
trainer.resume_from(
    "last",
    device="cuda",
)
```

---

# Citation

If you use Entangled State Attention in academic work, please cite the ESA preprint.

```bibtex
@misc{hussain2026entangledstateattention,
  title        = {Entangled State Attention: Associative State Scanning for Causal Sequence Modeling},
  author       = {Hussain, Zameer},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20973958}
}
```

Preprint DOI:

```text
https://doi.org/10.5281/zenodo.20973958
```

---

# Authors and Contributors

**Zameer Hussain**
Lead author and project developer.

**Akhtar Hussain**
Contributor to associative scan methodology and ESA optimization work.

---

# License

Apache License 2.0.

See the repository `LICENSE` file for details.

---

# Project Goal

The goal of Entangled State Attention is to investigate whether causal language models can achieve useful sequence modeling through efficient state evolution rather than relying exclusively on full token-to-token attention.

```text
Less compute.
More access.
Better AI.
```
