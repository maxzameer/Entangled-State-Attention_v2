# Copyright 2026 Zameer Hussain and Akhtar Hussain
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Backend = Literal["flare", "thunder", "pulse"]
Precision = Literal["fp8", "fp16", "bf16", "fp32", "fp64"]


@dataclass
class ESAConfig:
    embd: int
    head: int = 4
    batch: int | None = None
    block: int | None = None
    dropout: float = 0.0
    backend: Backend = "flare"
    compass: int | None = None
    precision: Precision = "fp16"
    gate_min: float = 0.80
    gate_max: float = 0.995
    eps: float = 1e-5
    strict_precision: bool = False
    strict_backend: bool = False
