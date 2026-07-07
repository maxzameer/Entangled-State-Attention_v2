from __future__ import annotations

import torch
import torch.nn as nn


class ESA(nn.Module):
    """
    Public ESA wrapper.

    Clean user API:

        from esa import ESA

        layer = ESA(
            embd=128,
            head=4,
            batch=16,
            block=1024,
            backend="thunder",
        )

    Internal Thunder defaults:
        compass=16
        gate_min=0.80
        gate_max=0.995
        eps=1e-5
    """

    def __init__(
        self,
        embd: int | None = None,
        head: int = 4,
        batch: int | None = None,
        block: int | None = None,
        backend: str = "thunder",
        precision: str = "fp16",

        # Backward-compatible aliases
        n_embd: int | None = None,
        n_head: int | None = None,
        batch_size: int | None = None,
        block_size: int | None = None,

        # Internal / expert options
        compass: int = 16,
        gate_min: float = 0.80,
        gate_max: float = 0.995,
        eps: float = 1e-5,

        # Convenience
        device: str | torch.device | None = "auto",
        auto_compile: bool = False,
        compile_mode: str = "reduce-overhead",

        # Speed / safety controls
        auto_move_input: bool = True,
        strict_checks: bool = False,
    ):
        super().__init__()

        # ----------------------------------------------------
        # Resolve new + old argument names
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
                "ESA requires embd, for example: ESA(embd=128, head=4)."
            )

        if embd % head != 0:
            raise ValueError(
                f"embd must be divisible by head. Got embd={embd}, head={head}."
            )

        if compass <= 0:
            raise ValueError(
                f"compass must be positive. Got compass={compass}."
            )

        # ----------------------------------------------------
        # Public attributes
        # ----------------------------------------------------
        self.embd = int(embd)
        self.head = int(head)
        self.batch = batch
        self.block = block

        # Backward-compatible attributes
        self.n_embd = self.embd
        self.n_head = self.head
        self.batch_size = batch
        self.block_size = block

        self.backend = str(backend).lower()
        self.precision = precision

        self.compass = int(compass)
        self.gate_min = float(gate_min)
        self.gate_max = float(gate_max)
        self.eps = float(eps)

        self.auto_move_input = bool(auto_move_input)
        self.strict_checks = bool(strict_checks)
        self.compiled = False

        # ----------------------------------------------------
        # Build backend
        # ----------------------------------------------------
        if self.backend == "thunder":
            from .backends.thunder import ThunderESA

            self.layer = ThunderESA(
                embd=self.embd,
                head=self.head,
                compass=self.compass,
                precision=self.precision,
                gate_min=self.gate_min,
                gate_max=self.gate_max,
                eps=self.eps,
            )

        elif self.backend == "flare":
            from .backends.flare import FlareESA

            self.layer = FlareESA(
                n_embd=self.embd,
                n_head=self.head,
                precision=self.precision,
                gate_min=self.gate_min,
                gate_max=self.gate_max,
                eps=self.eps,
            )

        elif self.backend == "pulse":
            from .backends.pulse import PulseESA

            self.layer = PulseESA(
                n_embd=self.embd,
                n_head=self.head,
                precision=self.precision,
                gate_min=self.gate_min,
                gate_max=self.gate_max,
                eps=self.eps,
            )

        else:
            raise ValueError(
                f"Unknown ESA backend: {backend}. "
                "Supported backends: 'thunder', 'flare', 'pulse'."
            )

        # ----------------------------------------------------
        # Fast device reference
        # ----------------------------------------------------
        # This avoids calling next(self.parameters()).device every forward.
        # The buffer automatically moves when the module moves with .to(device).
        self.register_buffer(
            "_esa_device_ref",
            torch.empty(0),
            persistent=False,
        )

        # ----------------------------------------------------
        # Automatic device placement
        # ----------------------------------------------------
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if device is not None:
            self.to(device)

        # ----------------------------------------------------
        # Optional compile
        # ----------------------------------------------------
        if auto_compile:
            self.compile(mode=compile_mode)

    def compile(self, mode: str = "reduce-overhead"):
        """
        Compile the internal ESA backend.

        Usage:
            layer = ESA(embd=128, head=4).compile()
        """
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
        # ----------------------------------------------------
        # Fast auto-move input
        # ----------------------------------------------------
        if self.auto_move_input:
            target_device = self._esa_device_ref.device

            if x.device != target_device:
                x = x.to(target_device, non_blocking=True)

        # ----------------------------------------------------
        # Optional strict checks
        # Disabled by default for speed.
        # Enable with strict_checks=True for debugging.
        # ----------------------------------------------------
        if self.strict_checks:
            if x.dim() != 3:
                raise ValueError(
                    f"ESA input must have shape [batch, seq, embd]. "
                    f"Got {tuple(x.shape)}."
                )

            B, T, C = x.shape

            if C != self.embd:
                raise ValueError(
                    f"ESA expected embedding dim {self.embd}, got {C}."
                )

            if self.batch is not None and B != self.batch:
                raise ValueError(
                    f"ESA expected batch {self.batch}, got {B}."
                )

            if self.block is not None and T > self.block:
                raise ValueError(
                    f"ESA block limit is {self.block}, got sequence length {T}."
                )

        return self.layer(x)