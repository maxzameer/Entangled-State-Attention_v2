from __future__ import annotations

from esa import compass, ESA


# This demo uses a fake evaluate_fn to show the API.
# In a real project, evaluate_fn should run a short training/evaluation window
# and return val_loss/ppl plus throughput.
def evaluate_fn(*, backend: str, c: int | None, precision: str):
    if backend == "pulse":
        return {"backend": "pulse", "c": None, "val_loss": 1.88, "tok_per_sec": 500_000}

    # Example Thunder results.
    table = {
        8: {"val_loss": 1.91, "tok_per_sec": 880_000},
        16: {"val_loss": 1.879, "tok_per_sec": 940_000},
        32: {"val_loss": 1.895, "tok_per_sec": 1_080_000},
        64: {"val_loss": 1.96, "tok_per_sec": 1_200_000},
    }
    return {"backend": backend, "c": c, **table[c]}


result = compass(
    evaluate_fn=evaluate_fn,
    c_candidates=(8, 16, 32, 64),
    reference_backend="pulse",
    precision="fp16",
    quality_tolerance=0.02,
)

print(result)
print(result.summary())

layer = ESA(n_embd=128, n_head=4, backend="thunder", c=result.recommended)
print(layer)
