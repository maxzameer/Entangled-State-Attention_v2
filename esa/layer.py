# Copyright 2026 Zameer Hussain and Akhtar Hussain
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import torch
import torch.nn as nn

from .config import ESAConfig
from .constants import SUPPORTED_BACKENDS
from .backends import ThunderESA, FlareESA, PulseESA


class ESA(nn.Module):
    """Entangled State Attention v2 wrapper layer.

    Default:
        ``ESA(n_embd=128)`` -> backend="thunder", c=16, precision="fp16".

    Backend names:
        - thunder: optimized chunked ESA backend. Supports ``c``.
        - flare: experimental Triton ESA backend. Does not expose ``c``.
        - pulse: base/reference ESA backend. Does not expose ``c``.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int = 4,
        dropout: float = 0.0,
        backend: str = "thunder",
        c: int | None = None,
        precision: str = "fp16",
        gate_min: float = 0.80,
        gate_max: float = 0.995,
        eps: float = 1e-6,
        strict_precision: bool = False,
        strict_backend: bool = False,
    ):
        super().__init__()
        backend = backend.lower()
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unknown backend={backend!r}. Expected one of {sorted(SUPPORTED_BACKENDS)}."
            )

        if backend == "thunder":
            if c is None:
                c = 16
            self.backend_layer = ThunderESA(
                n_embd=n_embd,
                n_head=n_head,
                dropout=dropout,
                c=c,
                precision=precision,
                gate_min=gate_min,
                gate_max=gate_max,
                eps=eps,
                strict_precision=strict_precision,
            )
        else:
            if c is not None:
                raise ValueError(
                    'The chunked scan parameter c is only supported by backend="thunder". '
                    'backend="flare" and backend="pulse" do not expose c.'
                )

            cls = FlareESA if backend == "flare" else PulseESA
            self.backend_layer = cls(
                n_embd=n_embd,
                n_head=n_head,
                dropout=dropout,
                precision=precision,
                gate_min=gate_min,
                gate_max=gate_max,
                eps=eps,
                strict_precision=strict_precision,
            )

        self.n_embd = n_embd
        self.n_head = n_head
        self.dropout = dropout
        self.backend = backend
        self.c = c if backend == "thunder" else None
        self.precision = precision
        self.strict_backend = strict_backend

    @classmethod
    def from_config(cls, config: ESAConfig) -> "ESA":
        return cls(
            n_embd=config.n_embd,
            n_head=config.n_head,
            dropout=config.dropout,
            backend=config.backend,
            c=config.c,
            precision=config.precision,
            gate_min=config.gate_min,
            gate_max=config.gate_max,
            eps=config.eps,
            strict_precision=config.strict_precision,
            strict_backend=config.strict_backend,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backend_layer(x)
