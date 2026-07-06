# Copyright 2026 Zameer Hussain and Akhtar Hussain
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Backend = Literal["thunder", "flare", "pulse"]
Precision = Literal["fp8", "fp16", "bf16", "fp32", "fp64"]


@dataclass
class ESAConfig:
    """Configuration for the ESA wrapper layer.

    Parameters
    ----------
    n_embd:
        Embedding dimension.
    n_head:
        Number of ESA heads. ``n_embd`` must be divisible by ``n_head``.
    backend:
        ``"thunder"`` is the optimized chunked ESA backend and is the default.
        ``"flare"`` is the experimental Triton backend.
        ``"pulse"`` is the base/reference backend.
    c:
        Thunder's chunked scan parameter. Only valid for backend="thunder".
        If None and backend="thunder", ESA uses c=16.
    precision:
        Numeric mode. fp16 is the default. fp8 is experimental. fp64 is for
        correctness/reference checks.
    """

    n_embd: int
    n_head: int = 4
    dropout: float = 0.0
    backend: Backend = "thunder"
    c: int | None = None
    precision: Precision = "fp16"
    gate_min: float = 0.80
    gate_max: float = 0.995
    eps: float = 1e-5
    strict_precision: bool = False
    strict_backend: bool = False
