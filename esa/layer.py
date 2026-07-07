# Copyright 2026 Zameer Hussain and Akhtar Hussain
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations


import torch
import torch.nn as nn

from .config import ESAConfig
from .constants import SUPPORTED_BACKENDS
from .backends import ThunderESA, FlareESA, PulseESA

import torch
import torch.nn as nn


class ESA(nn.Module):
    def __init__(
        self,

        # Clean public names
        embd: int | None = None,
        head: int = 4,
        batch: int | None = None,
        block: int | None = None,

        # Backend
        backend: str = "thunder",
        precision: str = "fp16",

        # Old names kept for backward compatibility
        n_embd: int | None = None,
        n_head: int | None = None,
        batch_size: int | None = None,
        block_size: int | None = None,

        # Internal ESA algorithm defaults
        compass: int = 16,
        gate_min: float = 0.80,
        gate_max: float = 0.995,
        eps: float = 1e-5,

        # Runtime
        device: str | torch.device | None = "auto",
        auto_compile: bool = False,
        compile_mode: str = "reduce-overhead",
    ):
        super().__init__()

        # ----------------------------------------------------
        # Resolve new names and old names
        # ----------------------------------------------------
        if embd is None:
            embd = n_embd

        if n_head is not None:
            head = n_head

        if batch is None:
            batch = batch_size

        if block is None:
            block = block_size

        if embd is None:
            raise ValueError(
                "ESA requires embd. Example: ESA(embd=128, head=4)"
            )

        if embd % head != 0:
            raise ValueError(
                f"embd must be divisible by head, got embd={embd}, head={head}"
            )

        # Clean public attributes
        self.embd = embd
        self.head = head
        self.batch = batch
        self.block = block

        # Backward-compatible attributes
        self.n_embd = embd
        self.n_head = head
        self.batch_size = batch
        self.block_size = block

        self.backend = backend
        self.precision = precision

        # Internal algorithm constants
        self.compass = compass
        self.gate_min = gate_min
        self.gate_max = gate_max
        self.eps = eps

        # ----------------------------------------------------
        # Build backend
        # ----------------------------------------------------
        if backend == "thunder":
            from .backends.thunder import ThunderESA

            self.layer = ThunderESA(
                embd=embd,
                head=head,
                compass=compass,
                precision=precision,
                gate_min=gate_min,
                gate_max=gate_max,
                eps=eps,
            )

        elif backend == "flare":
            from .backends.flare import FlareESA

            self.layer = FlareESA(
                n_embd=embd,
                n_head=head,
                precision=precision,
                gate_min=gate_min,
                gate_max=gate_max,
                eps=eps,
            )

        elif backend == "pulse":
            from .backends.pulse import PulseESA

            self.layer = PulseESA(
                n_embd=embd,
                n_head=head,
                precision=precision,
                gate_min=gate_min,
                gate_max=gate_max,
                eps=eps,
            )

        else:
            raise ValueError(
                f"Unknown ESA backend: {backend}. "
                f"Expected 'thunder', 'flare', or 'pulse'."
            )

        # ----------------------------------------------------
        # Auto device
        # ----------------------------------------------------
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if device is not None:
            self.to(device)

        # ----------------------------------------------------
        # Optional compile
        # ----------------------------------------------------
        self.compiled = False

        if auto_compile:
            self.compile(mode=compile_mode)

    def compile(self, mode: str = "reduce-overhead"):
        try:
            self.layer = torch.compile(
                self.layer,
                mode=mode,
                fullgraph=False,
            )
            self.compiled = True
        except Exception:
            self.compiled = False

        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Auto move input to same device as ESA layer
        param_device = next(self.parameters()).device

        if x.device != param_device:
            x = x.to(param_device)

        if x.dim() != 3:
            raise ValueError(
                f"ESA expects input shape [batch, block, embd], got {tuple(x.shape)}"
            )

        B, T, C = x.shape

        if C != self.embd:
            raise ValueError(
                f"Last dimension must match embd={self.embd}, got {C}"
            )

        if self.batch is not None and B != self.batch:
            raise ValueError(
                f"Expected batch={self.batch}, got batch={B}"
            )

        if self.block is not None and T > self.block:
            raise ValueError(
                f"Input block length {T} exceeds configured block={self.block}"
            )

        return self.layer(x)