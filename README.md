# Entangled State Attention v2.1.1

**Entangled State Attention (ESA)** is a state-based causal sequence modeling architecture for efficient full-sequence training and recurrent text generation.

Instead of constructing an explicit token-to-token attention score matrix, ESA updates a causal state through the recurrence

$$
E_t = A_t \odot E_{t-1} + B_t
$$

ESA v2.1.1 provides both a reusable PyTorch sequence layer and a complete causal language model.

---

## ESA v2.1.1 at a glance

* `ESA` — reusable PyTorch sequence-mixing layer
* `Thunder C16` — default ESA backend
* `Thunder` — optimized associative scan backend with configurable `compass`
* `Pulse` — reference backend for correctness and comparison
* `Flare` — alternative full-sequence backend
* `ESAModel` — complete causal language model built from ESA layers
* `ESA-Lightning` — recurrent generation engine
* `Trainer` — checkpoint management and exact training resume
* `compass()` — utility for selecting practical Thunder scan settings
* `thunderBoost()` — optional compile and warmup utility

> **Important:** ESA-Lightning is a generation runtime, not a selectable public training backend.
>
> Public ESA backends are `thunder`, `pulse`, and `flare`.

---

# Optimized defaults

ESA v2.1.1 uses optimized defaults for the common path:

* **Default ESA backend:** Thunder
* **Default Thunder compass:** `16`
* **Default model backend:** Thunder
* **Default training precision mode:** `fp16`
* **Default generation prefill:** `thunder_16`
* **Default autoregressive runtime:** ESA-Lightning
* **Optional compiled prefill:** `thunder_compiled_16`
* **Optional training compilation:** `torch.compile(..., mode="reduce-overhead", fullgraph=False)`

Normal text generation is intentionally simple:

```python
text = model.generate(
    "Once upon a time",
    tokenizer=tokenizer,
    seek=256,
)
```

Advanced users can select a different prefill engine:

```python
text = model.generate(
    "Once upon a time",
    tokenizer=tokenizer,
    seek=256,
    prefill="thunder_compiled_32",
    runtime="lightning",
)
```

`max_new_tokens` remains accepted as a backward-compatible alias for `seek`.

---

# Installation

## Requirements

```text
Python >= 3.10
PyTorch >= 2.1
```

The Python package is imported as:

```python
import esa
```

The distribution name is:

```text
entangled-state-attention
```

---

## Install ESA v2.1.1 from the GitHub test branch

```bash
pip install --no-cache-dir git+https://github.com/maxzameer/Entangled-State-Attention_v2.git@test/v2.1.1
```

---

## Local development install

```bash
git clone https://github.com/maxzameer/Entangled-State-Attention_v2.git
cd Entangled-State-Attention_v2
git checkout test/v2.1.1
pip install -e .
```

---

## PyPI installation

After ESA v2.1.1 is published to PyPI:

```bash
pip install entangled-state-attention==2.1.1
```

---

## Check the installed version

```python
import esa

print(esa.__version__)
```

Expected:

```text
2.1.1
```

---

# 1. ESA layer

Use `ESA` when you want to integrate Entangled State Attention into your own PyTorch architecture.

```python
import torch
from esa import ESA

layer = ESA(
    embd=128,
    head=4,
    batch=16,
    block=1024,
    backend="thunder",
    compass=16,
    precision="fp16",
)

x = torch.randn(
    16,
    1024,
    128,
    device="cuda",
)

y = layer(x)

print(y.shape)
```

Output:

```text
torch.Size([16, 1024, 128])
```

## Main arguments

```text
embd          embedding dimension
head          number of ESA heads
batch         optional expected batch size
block         optional expected sequence length
backend       "thunder", "pulse", or "flare"
precision     backend precision mode
compass       optional Thunder scan setting
dropout       dropout probability
device        "auto", "cpu", "cuda", or torch.device
auto_compile  optionally compile the layer
```

The canonical v2.1.1 public names are:

```text
embd
head
batch
block
compass
```

Old constructor names such as `n_embd`, `n_head`, and `c` are not part of the public `ESA` API.

---

# 2. Backends

## Thunder C16 — default backend

Thunder with `compass=16` is the default public ESA backend.

```python
from esa import ESA

layer = ESA(
    embd=128,
    head=4,
)
```

This is equivalent to the default backend configuration:

```python
layer = ESA(
    embd=128,
    head=4,
    backend="thunder",
    compass=16,
)
```

Thunder is the optimized associative scan backend and supports configurable `compass` values.

Example:

```python
layer = ESA(
    embd=128,
    head=4,
    backend="thunder",
    compass=32,
)
```

Most users can keep the default `compass=16`.

---

## Pulse — reference backend

```python
from esa import ESA

layer = ESA(
    embd=128,
    head=4,
    backend="pulse",
)
```

Pulse is useful for:

* correctness checks
* debugging
* backend comparisons
* recurrent equivalence tests
* research validation

`compass` is not supported for Pulse.

---

## Flare — alternative full-sequence backend

```python
from esa import ESA

layer = ESA(
    embd=128,
    head=4,
    backend="flare",
)
```

Flare is available as an alternative full-sequence ESA backend.

`compass` is not supported for Flare.

---

# 3. Precision and mixed precision

The public default precision mode is:

```python
precision="fp16"
```

Supported precision modes may include:

```text
fp16
bf16
fp32
fp64
fp8
```

Availability and behavior can depend on the selected backend and hardware.

## Recommended CUDA mixed-precision usage

ESA parameters can remain in FP32 while CUDA AMP performs mixed-precision computation.

```python
import torch
from esa import ESA

layer = ESA(
    embd=128,
    head=4,
    backend="thunder",
    precision="fp16",
    device="cuda",
)

x = torch.randn(
    8,
    512,
    128,
    device="cuda",
)

with torch.autocast(
    device_type="cuda",
    dtype=torch.float16,
):
    y = layer(x)
```

For direct layer usage, do not manually pass FP16 tensors to FP32 module weights outside an autocast context.

---

# 4. Complete ESA language model

ESA v2.1.1 includes `ESAModel`, a complete causal language model built from ESA layers.

```python
from esa import ESAModel, ESAModelConfig

config = ESAModelConfig(
    vocab_size=50257,
    block=1024,
    n_layer=6,
    head=6,
    embd=384,
    dropout=0.1,
    backend="thunder",
    precision="fp16",
    compass=16,
)

model = ESAModel(
    config,
    device="cuda",
)
```

The default `ESAModelConfig` architecture values are:

```text
block          512
n_layer        6
head           6
embd           384
dropout        0.1
backend        thunder
precision      fp16
compass        backend default
```

A model can also be created directly with keyword arguments:

```python
from esa import ESAModel

model = ESAModel(
    vocab_size=50257,
    block=1024,
    n_layer=6,
    head=6,
    embd=384,
    backend="thunder",
    precision="fp16",
    compass=16,
    device="cuda",
)
```

---

## Forward pass

Training with targets:

```python
logits, loss = model(
    input_ids,
    targets,
)
```

Inference without targets:

```python
logits, loss = model(
    input_ids,
)
```

The returned `loss` is `None` when targets are not provided.

---

# 5. Training compilation

`ESAModel` can compile the full training forward path:

```python
model.compile_training()
```

Or configure it explicitly:

```python
model.compile_training(
    mode="reduce-overhead",
    fullgraph=False,
)
```

The default model configuration includes:

```text
training_compile           True
training_compile_mode      reduce-overhead
training_compile_fullgraph False
```

Compilation behavior depends on the platform, PyTorch version, and available accelerator.

---

# 6. Text generation with ESA-Lightning

`ESAModel.generate()` is the main public generation API.

ESA-Lightning performs recurrent generation using ESA state instead of maintaining an attention KV cache.

```python
text = model.generate(
    "Entangled State Attention is",
    tokenizer=tokenizer,
    seek=256,
    temperature=0.8,
    top_k=50,
)
```

The prompt can be passed positionally:

```python
text = model.generate(
    "Once upon a time",
    tokenizer=tokenizer,
    seek=256,
)
```

Or explicitly:

```python
text = model.generate(
    prompt="Once upon a time",
    tokenizer=tokenizer,
    seek=256,
)
```

---

## Generate from token IDs

```python
output_ids = model.generate(
    input_ids=input_ids,
    seek=256,
    temperature=0.8,
    top_k=50,
)
```

Advanced token-level generation is also available through:

```python
output_ids = model.generate_ids(
    input_ids,
    seek=256,
)
```

---

## Generation arguments

```text
prompt             optional raw-text prompt
tokenizer          tokenizer used for raw-text generation
input_ids          optional token tensor
seek               number of tokens to generate
prefill            prefill execution engine
runtime            autoregressive runtime
temperature        sampling temperature
top_k              optional top-k sampling
top_p              optional nucleus sampling
eos_token_id       optional early-stop token
seed               optional random seed
compile            compile recurrent decode
compile_mode       torch.compile mode
progress_interval  optional progress reporting interval
stats              return GenerationResult with statistics
max_new_tokens     backward-compatible alias for seek
```

---

## Detailed generation statistics

Set:

```python
stats=True
```

Example:

```python
result = model.generate(
    "The future of efficient language models",
    tokenizer=tokenizer,
    seek=512,
    stats=True,
)

print(result.text)
print(result.generated_ids)
print(result.stats)
```

`GenerationStats` includes information such as:

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

# 7. Prefill and ESA-Lightning decode

The optimized generation path separates prompt prefill from recurrent one-token decoding.

## Prefill

```python
logits, states, position = model.prefill(
    input_ids,
    engine="thunder_16",
)
```

Compiled Thunder prefill:

```python
logits, states, position = model.prefill(
    input_ids,
    engine="thunder_compiled_16",
    compile_mode="reduce-overhead",
    fullgraph=False,
    dynamic=True,
)
```

---

## Recurrent decode step

```python
logits, next_states = model.lightning_step(
    token,
    states,
    pos_tensor,
)
```

The state shape is independent of the full prompt length and is updated recurrently during decoding.

---

## Backward-compatible prefill

The following interface is also available:

```python
logits, states, position = model.lightning_prefill(
    input_ids
)
```

This uses the model's configured backend.

For normal user-facing generation, prefer:

```python
model.generate(...)
```

---

# 8. Compile generation

The fixed-shape ESA-Lightning one-token decode step can be compiled:

```python
model.compile_generation()
```

Or:

```python
model.compile_generation(
    mode="reduce-overhead",
    fullgraph=False,
)
```

Generation can also compile the recurrent path automatically:

```python
text = model.generate(
    "Once upon a time",
    tokenizer=tokenizer,
    seek=256,
    compile=True,
)
```

Prefill and decode intentionally use different compilation strategies:

```text
Prefill
    dynamic prompt length
    dynamic compilation support

Decode
    fixed one-token shape
    reduce-overhead compilation
```

---

# 9. Save a trained model

Use the model lifecycle API:

```python
model.save(
    "my_esa_model"
)
```

The saved path is a model directory containing files such as:

```text
my_esa_model/
├── config.json
├── model.pt
└── metadata.json
```

Custom metadata can be included:

```python
model.save(
    "my_esa_model",
    metadata={
        "dataset": "English Wikipedia",
        "training_steps": 20000,
        "notes": "continued pretraining",
    },
)
```

`metadata.json` can contain information such as:

```text
architecture
generation_engine
backend
format_version
custom user metadata
```

---

# 10. Load a saved model

Load the model directory:

```python
from esa import ESAModel

model = ESAModel.load(
    "my_esa_model",
    device="cuda",
)
```

Strict loading is enabled by default:

```python
model = ESAModel.load(
    "my_esa_model",
    device="cuda",
    strict=True,
)
```

The loader reconstructs the model configuration and restores the model state.

---

# 11. Continue training a loaded model

```python
import torch
from esa import ESAModel

model = ESAModel.load(
    "my_esa_model",
    device="cuda",
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
)

model.train()

for input_ids, targets in train_loader:

    input_ids = input_ids.cuda()
    targets = targets.cuda()

    optimizer.zero_grad(
        set_to_none=True
    )

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
    ):
        logits, loss = model(
            input_ids,
            targets,
        )

    loss.backward()
    optimizer.step()
```

For production mixed-precision training, a gradient scaler can also be used where appropriate.

---

# 12. Trainer and training checkpoints

ESA v2.1.1 includes `Trainer` for checkpoint management and exact training resume.

`Trainer` is a checkpoint manager. It does not replace your normal PyTorch training loop.

```python
from esa import Trainer

trainer = Trainer(
    model,
    optimizer=optimizer,
    scheduler=scheduler,
    scaler=scaler,
    checkpoint_dir="checkpoints",
    save_every=1000,
    save_at=[5000, 10000, 20000],
    save_best=True,
    save_last=True,
    keep_last_n=3,
)
```

Supported checkpoint behavior:

```text
save_every=N       save periodically
save_at=[...]      save exact requested steps
save_best=True     maintain the best validation checkpoint
save_last=True     save the final/latest checkpoint
keep_last_n=N      retain only the latest periodic checkpoints
```

---

## Save automatically during training

```python
saved_paths = trainer.maybe_save(
    step=step,
    val_loss=val_loss,
)
```

---

## Save a named checkpoint

```python
checkpoint = trainer.save_checkpoint(
    step=5000,
    name="wikipedia_stage_1",
    protected=True,
)
```

Optional extra information can be stored:

```python
checkpoint = trainer.save_checkpoint(
    step=5000,
    name="wikipedia_stage_1",
    protected=True,
    extra={
        "dataset": "Wikipedia",
    },
)
```

---

## Save the final checkpoint

```python
trainer.save_final(
    step=final_step,
    val_loss=final_val_loss,
)
```

---

## Load a checkpoint

```python
payload = trainer.load_checkpoint(
    "checkpoints/step_010000",
)
```

By default, checkpoint loading can restore:

* model state
* optimizer state
* scheduler state
* gradient scaler state
* trainer step
* best validation loss
* Python RNG state
* PyTorch RNG state
* CUDA RNG state

RNG restoration can be controlled with:

```python
trainer.load_checkpoint(
    "checkpoints/step_010000",
    restore_rng=True,
)
```

---

# 13. Resume training

Resume the latest checkpoint:

```python
trainer.resume_from(
    "latest"
)
```

Resume the best checkpoint:

```python
trainer.resume_from(
    "best"
)
```

Resume a specific checkpoint:

```python
trainer.resume_from(
    "checkpoints/step_010000"
)
```

`resume_from()` restores the checkpoint into the existing `Trainer` and returns the trainer instance.

---

# 14. Model information

```python
info = model.model_info()

print(info)
```

The returned dictionary includes model configuration and runtime information such as:

```text
vocab_size
block
n_layer
head
embd
dropout
backend
precision
compass
parameters
device
generation_engine
```

The generation engine is reported as:

```text
ESA-Lightning
```

---

# 15. thunderBoost

`thunderBoost()` is an optional compile and warmup utility for ESA layers and general PyTorch modules.

```python
from esa import thunderBoost

model = thunderBoost(
    model,
    batch=batch,
)
```

To receive state/statistics information:

```python
model, state = thunderBoost(
    model,
    batch=batch,
    state=True,
)
```

The utility supports options such as:

```text
compile
compile_mode
backward
amp
dtype
device
steps
return_stats
```

`thunderBoost()` is separate from:

* normal model optimization
* ESA backend selection
* ESA-Lightning generation

---

# 16. Compass

`compass()` evaluates Thunder ESA across candidate scan settings and recommends a practical value for a workload.

```python
from esa import compass

result = compass(
    evaluate_fn=evaluate_fn,
    c_candidates=(8, 16, 32, 64),
    reference_backend="pulse",
    precision="fp16",
    quality_tolerance=0.02,
)

print(result.recommended)
print(result.summary())
```

The evaluation function receives arguments such as:

```python
def evaluate_fn(
    *,
    backend: str,
    c: int | None,
    precision: str,
):
    ...
```

`CompassResult` provides:

```text
recommended
best_quality
fastest
rows
reference_backend
precision
quality_tolerance
recommendation
summary()
```

`compass()` uses `c_candidates` as its public candidate-list argument.

The resulting recommended value is passed to `ESA` through the public `compass` argument:

```python
from esa import ESA

layer = ESA(
    embd=128,
    head=4,
    backend="thunder",
    compass=result.recommended,
)
```

---

# 17. Public API

```python
from esa import (
    ESA,
    ESAConfig,
    ESAModel,
    ESAModelConfig,
    Trainer,
    TrainerState,
    GenerationResult,
    GenerationStats,
    FlareESA,
    ThunderESA,
    PulseESA,
    compass,
    CompassResult,
    thunderBoost,
)
```

---

# 18. Recommended high-level workflow

## Build

```python
from esa import ESAModel

model = ESAModel(
    vocab_size=vocab_size,
    backend="thunder",
    compass=16,
    device="cuda",
)
```

---

## Train

```python
logits, loss = model(
    input_ids,
    targets,
)
```

---

## Generate

```python
text = model.generate(
    prompt,
    tokenizer=tokenizer,
    seek=256,
)
```

---

## Save

```python
model.save(
    "my_model"
)
```

---

## Load

```python
model = ESAModel.load(
    "my_model",
    device="cuda",
)
```

---

## Resume exact training state

```python
trainer.resume_from(
    "latest"
)
```

---

# 19. Architecture roles

ESA v2.1.1 separates full-sequence execution and recurrent generation clearly:

```text
Full-sequence ESA execution
    Thunder C16    default
    Pulse          reference
    Flare          alternative

Text generation
    Prefill
        Thunder eager or compiled
        selectable execution engine

    ESA-Lightning
        recurrent state
        one-token decode
        model.generate()
```

ESA-Lightning is not exposed as:

```python
backend="lightning"
```

The public backend names are:

```text
thunder
pulse
flare
```

---

# 20. Validated v2.1.1 lifecycle

ESA v2.1.1 has been tested across the following lifecycle:

```text
package import
public API
official examples
Thunder forward/backward
Pulse forward/backward
Flare forward/backward
FP32 execution
CUDA AMP execution
recurrent prefill
one-token Lightning decode
model training
loss reduction
token generation
raw-text generation
compiled prefill
compiled recurrent generation
model save
model load
exact output roundtrip
checkpoint save
RNG restoration
optimizer resume
exact training continuation
wheel build
source distribution build
external wheel installation
external source distribution installation
```

Release validation confirmed exact save/load output equivalence and exact checkpoint continuation in the tested configuration.

---


<!-- ESA_COMPILE_MODE_GUIDE_START -->
# Compile modes

ESA keeps compilation enabled by default using PyTorch's standard `default`
compile mode. This provides strong acceleration while prioritizing stability
and compatibility.

## Default behavior

Training:

```python
config = ESAModelConfig(
    vocab_size=50257,
    training_compile=True,
    training_compile_mode="default",
)
```

Generation:

```python
text = model.generate(
    "Once upon a time",
    tokenizer=tokenizer,
    seek=256,
    compile=True,
)
```

Equivalent explicit generation configuration:

```python
text = model.generate(
    "Once upon a time",
    tokenizer=tokenizer,
    seek=256,
    compile=True,
    compile_mode="default",
)
```

## Optimum performance

For optimum CUDA performance, manually enable `reduce-overhead` after
validating the exact model, device, batch size, tensor shapes, recurrent
state, and memory budget used by the application.

### Generation

```python
text = model.generate(
    "Once upon a time",
    tokenizer=tokenizer,
    seek=256,
    compile=True,
    compile_mode="reduce-overhead",
)
```

### Training

```python
config = ESAModelConfig(
    vocab_size=50257,
    training_compile=True,
    training_compile_mode="reduce-overhead",
)
```

### Compiled prefill

```python
logits, states, position = model.prefill(
    input_ids,
    engine="thunder_compiled_16",
    compile_mode="reduce-overhead",
    fullgraph=False,
    dynamic=True,
)
```

### Compile the recurrent decode step directly

```python
model.compile_generation(
    mode="reduce-overhead",
    fullgraph=False,
)
```

> **Recommendation:** use the default compile mode for general applications.
> Manually enable `reduce-overhead` when maximum CUDA performance is required
> and the exact workload has been tested for stability.
<!-- ESA_COMPILE_MODE_GUIDE_END -->

---

# Research status

ESA v2.1.1 is research software.

Performance depends on:

* model size
* sequence length
* batch size
* precision
* selected backend
* Thunder compass setting
* GPU architecture
* PyTorch version
* compilation mode
* dataset
* training configuration

Benchmark results should be reported with complete hardware and configuration details.

---

# Citation

If you use Entangled State Attention in academic work, please cite:

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
Hussain, Z., & Hussain, A. (2026).
Entangled State Attention: Efficient Long-Context Causal Modeling via Associative State Scans (2.0.0).
Zenodo.
https://doi.org/10.5281/zenodo.21218821
```

The research-paper version and the Python library version are versioned independently.

---

# License

Apache License 2.0.

Copyright 2026 Zameer Hussain and Akhtar Hussain.
