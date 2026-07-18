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

    # Optimized ESA defaults
    backend: Backend = "thunder"
    compass: int | None = None

    precision: Precision = "fp16"
    gate_min: float = 0.8
    gate_max: float = 0.995
    eps: float = 1e-5
    strict_precision: bool = False
    strict_backend: bool = False

    def __post_init__(self) -> None:
        if self.embd <= 0:
            raise ValueError(f"embd must be positive, got {self.embd}.")

        if self.head <= 0:
            raise ValueError(f"head must be positive, got {self.head}.")

        if self.embd % self.head != 0:
            raise ValueError(
                f"embd must be divisible by head, "
                f"got embd={self.embd}, head={self.head}."
            )

        if self.backend not in {"thunder", "pulse", "flare"}:
            raise ValueError(
                f"Unknown ESA backend: {self.backend!r}. "
                "Supported backends: 'thunder', 'pulse', 'flare'."
            )

        # Thunder uses compass=16 when the user does not specify it.
        if self.backend == "thunder" and self.compass is None:
            self.compass = 16

        # Compass is only meaningful for Thunder.
        if self.backend != "thunder" and self.compass is not None:
            raise ValueError(
                "compass is only supported by backend='thunder'."
            )