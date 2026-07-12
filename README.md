# Entangled State Attention v2.1

**Entangled State Attention (ESA)** is a state-based causal sequence modeling architecture for efficient training and recurrent text generation.

Instead of constructing an explicit token-to-token attention score matrix, ESA updates a causal state through the recurrence

$$
E_t = A_t \odot E_{t-1} + B_t
$$

ESA v2.1 provides both a reusable sequence layer and a complete causal language model.

## ESA v2.1 at a glance

- `ESA` — reusable PyTorch sequence layer
- `Flare` — default high-performance training backend
- `Thunder` — optimized alternative backend
- `Pulse` — reference backend for correctness and comparison
- `ESAModel` — complete causal language model built from ESA layers
- `ESA-Lightning` — recurrent generation engine used by `model.generate()`
- `Trainer` — checkpointing and exact training resume
- `compass()` — utility for selecting practical Thunder scan settings
- `thunderBoost()` — optional compile and warmup utility

> **Important:** ESA-Lightning is a generation engine, not a selectable training backend.  
> Public training backends are `flare`, `thunder`, and `pulse`.

---

## Installation

### Install ESA v2.1 from the update branch

```bash
pip install --no-cache-dir git+https://github.com/maxzameer/Entangled-State-Attention_v2.git@main
```

### Local development install

```bash
git clone https://github.com/maxzameer/Entangled-State-Attention_v2.git
cd Entangled-State-Attention_v2
git checkout esa-v2.1-update
pip install -e .
```

### With optional Triton dependency

```bash
pip install -e .[triton]
```

Package version:

```python
import esa
print(esa.__version__)
```

Expected:

```text
2.1.0
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
    backend="flare",
    precision="fp16",
)

x = torch.randn(16, 1024, 128, device="cuda")
y = layer(x)

print(y.shape)
```

Output:

```text
torch.Size([16, 1024, 128])
```

## Main arguments

```text
embd       embedding dimension
head       number of ESA heads
batch      expected batch size
block      expected sequence length
backend    "flare", "thunder", or "pulse"
precision  scan precision mode
compass    optional Thunder scan setting
```

---

# 2. Backends

## Flare — default backend

Flare is the default ESA v2.1 training backend.

```python
from esa import ESA

layer = ESA(
    embd=128,
    head=4,
    block=1024,
    backend="flare",
    precision="fp16",
)
```

Flare is intended for high-performance GPU training.

## Thunder — optimized alternative

```python
from esa import ESA

layer = ESA(
    embd=128,
    head=4,
    block=1024,
    backend="thunder",
    precision="fp16",
)
```

Thunder supports the `compass` scan setting:

```python
layer = ESA(
    embd=128,
    head=4,
    block=1024,
    backend="thunder",
    compass=16,
)
```

Most users do not need to choose a `compass` value manually.

## Pulse — reference backend

```python
from esa import ESA

layer = ESA(
    embd=128,
    head=4,
    block=1024,
    backend="pulse",
)
```

Pulse is useful for:

- correctness checks
- debugging
- backend comparisons
- research validation

---

# 3. Precision

Default precision:

```python
precision="fp16"
```

Supported precision modes:

```text
fp16  default training mode
bf16  optional mixed-precision mode
fp32  stability and debugging mode
fp64  high-precision reference mode
fp8   experimental mode
```

---

# 4. Complete ESA language model

ESA v2.1 includes `ESAModel`, a complete causal language model built directly from ESA layers.

```python
from esa import ESAModel, ESAModelConfig

config = ESAModelConfig(
    vocab_size=50257,
    block=1024,
    n_layer=6,
    head=6,
    embd=384,
    dropout=0.1,
    backend="flare",
    precision="fp16",
)

model = ESAModel(config).cuda()
```

The default `ESAModelConfig` architecture values are:

```text
block      512
n_layer    6
head       6
embd       384
dropout    0.1
backend    flare
precision  fp16
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
    backend="flare",
    precision="fp16",
)
```

## Forward pass

```python
logits, loss = model(input_ids, targets)
```

For inference without targets:

```python
logits, loss = model(input_ids)
```

---

# 5. Text generation with ESA-Lightning

`ESAModel.generate()` is the public generation API.

ESA-Lightning performs recurrent generation using ESA state instead of building an attention KV cache.

```python
text = model.generate(
    prompt="Entangled State Attention is",
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.8,
    top_k=50,
)
```

You can also generate directly from token IDs:

```python
output_ids = model.generate(
    input_ids=input_ids,
    max_new_tokens=256,
    temperature=0.8,
    top_k=50,
)
```

## Generation arguments

```text
input_ids        optional token tensor
prompt           optional text prompt
tokenizer        tokenizer used with prompt=
max_new_tokens   number of tokens to generate
temperature      sampling temperature
top_k            optional top-k sampling
top_p            optional nucleus sampling
eos_token_id     optional early-stop token
seed             optional sampling seed
compile          compile recurrent generation step
compile_mode     torch.compile mode
progress_every   optional progress interval
return_result    return detailed GenerationResult
```

Example with detailed statistics:

```python
result = model.generate(
    prompt="The future of efficient language models",
    tokenizer=tokenizer,
    max_new_tokens=512,
    return_result=True,
)

print(result.text)
print(result.stats)
```

## ESA-Lightning low-level interface

Advanced deployment and export workflows can use:

```python
logits, states, position = model.lightning_prefill(input_ids)
logits, next_states = model.lightning_step(token, states, pos_tensor)
```

These methods are useful for recurrent inference and deployment targets such as ExecuTorch.

For normal text generation, use:

```python
model.generate(...)
```

---

# 6. Compile generation

The ESA-Lightning recurrent decode step can be compiled:

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

`model.generate()` can also compile the generation path automatically:

```python
text = model.generate(
    prompt="Once upon a time",
    tokenizer=tokenizer,
    max_new_tokens=256,
    compile=True,
)
```

---

# 7. Save a trained model

Use the model lifecycle API:

```python
model.save("my_esa_model")
```

The saved directory contains:

```text
my_esa_model/
├── config.json
├── model.pt
└── metadata.json
```

You can include custom metadata:

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

`metadata.json` records information such as:

```text
architecture       ESAModel
generation_engine  ESA-Lightning
backend            model training backend
format_version     model format version
```

---

# 8. Load a saved model

Load the model directory, not the individual `.pt` file:

```python
from esa import ESAModel

model = ESAModel.load(
    "my_esa_model",
    device="cuda",
)
```

The loader reconstructs the model from:

```text
config.json
model.pt
```

`metadata.json` is informational and is not required to reconstruct the model.

Strict loading is enabled by default:

```python
model = ESAModel.load(
    "my_esa_model",
    device="cuda",
    strict=True,
)
```

---

# 9. Continue training a loaded model

```python
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

    optimizer.zero_grad(set_to_none=True)

    logits, loss = model(
        input_ids,
        targets,
    )

    loss.backward()
    optimizer.step()
```

---

# 10. Training checkpoints

ESA v2.1 includes `Trainer` for checkpoint management and exact resume.

```python
from esa import Trainer

trainer = Trainer(
    model,
    optimizer=optimizer,
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
save_at=[...]      preserve exact requested checkpoints
save_best=True     save best validation checkpoint
save_last=True     save final/latest checkpoint
keep_last_n=N      retain only the latest periodic checkpoints
```

## Save during training

```python
saved_paths = trainer.maybe_save(
    step=step,
    val_loss=val_loss,
)
```

## Save a named checkpoint

```python
trainer.save_checkpoint(
    step=5000,
    name="wikipedia_stage_1",
    protected=True,
)
```

## Save the final checkpoint

```python
trainer.save_final()
```

A training checkpoint contains the model files plus:

```text
training_state.pt
```

The training state can include:

- current step
- best validation loss
- optimizer state
- scheduler state
- gradient scaler state
- Python RNG state
- PyTorch RNG state
- CUDA RNG state

---

# 11. Resume training

Resume the latest checkpoint:

```python
trainer.resume_from("latest")
```

Resume the best checkpoint:

```python
trainer.resume_from("best")
```

Resume a specific checkpoint:

```python
trainer.resume_from(
    "checkpoints/step_010000"
)
```

Or load a checkpoint directly:

```python
trainer.load_checkpoint(
    "checkpoints/step_010000"
)
```

This restores the model and, when available, optimizer, scheduler, scaler, and RNG state.

---

# 12. Model information

```python
info = model.model_info()
print(info)
```

The returned dictionary includes the model configuration together with information such as:

```text
parameters
device
generation_engine = ESA-Lightning
```

---

# 13. thunderBoost

`thunderBoost()` is an optional compile and warmup utility for ESA layers and PyTorch models.

It performs warmup without calling `optimizer.step()`, so it does not update model weights.

```python
from esa import thunderBoost

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
```

`thunderBoost()` is separate from normal ESA model training and from ESA-Lightning generation.

---

# 14. Compass

`compass()` evaluates Thunder ESA across candidate scan settings and recommends a practical value for a workload.

```python
from esa import compass

result = compass(
    evaluate_fn=evaluate_fn,
    compass_candidates=(8, 16, 32, 64),
    reference_backend="pulse",
    precision="fp16",
    quality_tolerance=0.02,
)

print(result.recommended)
print(result.summary())
```

`CompassResult` provides:

```text
recommended
best_quality
fastest
rows
summary()
```

---

# 15. Public API

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

## Recommended high-level workflow

### Build

```python
model = ESAModel(
    vocab_size=vocab_size,
    backend="flare",
)
```

### Train

```python
logits, loss = model(input_ids, targets)
```

### Generate

```python
text = model.generate(
    prompt=prompt,
    tokenizer=tokenizer,
    max_new_tokens=256,
)
```

### Save

```python
model.save("my_model")
```

### Load

```python
model = ESAModel.load(
    "my_model",
    device="cuda",
)
```

### Resume exact training state

```python
trainer.resume_from("latest")
```

---

# 16. Architecture roles

ESA v2.1 separates training and generation responsibilities clearly:

```text
Training / full-sequence execution
    Flare      default
    Thunder    optimized alternative
    Pulse      reference

Text generation
    ESA-Lightning
        prefill
        recurrent state
        one-token decode step
        model.generate()
```

ESA-Lightning is not exposed as `backend="lightning"`.

---

# Research status

ESA v2.1 is research software.

Performance depends on:

- model size
- sequence length
- batch size
- precision
- GPU architecture
- dataset
- training configuration

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
Zenodo. https://doi.org/10.5281/zenodo.21218821
```

---

# License

Apache License 2.0.

Copyright 2026 Zameer Hussain and Akhtar Hussain.
