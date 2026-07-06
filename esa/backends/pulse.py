# Copyright 2026 Zameer Hussain and Akhtar Hussain
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import torch
import torch.nn as nn

from ..precision import resolve_scan_dtype


def pulse_scan(A: torch.Tensor, B_write: torch.Tensor) -> torch.Tensor:
    """Pulse direct causal ESA recurrence.

    ``E_t = A_t * E_{t-1} + B_t``

    This is a reference implementation. It is simple and useful for debugging
    and correctness comparison, but it is not intended to be the fastest path.
    """
    if A.shape != B_write.shape:
        raise ValueError(f"A and B_write must have same shape, got {A.shape} and {B_write.shape}")
    if A.dim() != 4:
        raise ValueError(f"expected A/B_write shape [B,T,H,D], got {A.shape}")

    Bsz, T, H, D = A.shape
    state = B_write.new_zeros(Bsz, H, D)
    outputs = []
    for t in range(T):
        state = A[:, t] * state + B_write[:, t]
        outputs.append(state)
    return torch.stack(outputs, dim=1)


class PulseESA(nn.Module):
    """Pulse backend: base/reference ESA backend.

    Pulse does not expose ``c``. Use it for correctness checks, comparison,
    or debugging.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int = 4,
        dropout: float = 0.0,
        precision: str = "fp16",
        gate_min: float = 0.80,
        gate_max: float = 0.995,
        eps: float = 1e-6,
        strict_precision: bool = False,
    ):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.precision = precision
        self.gate_min = gate_min
        self.gate_max = gate_max
        self.eps = eps
        self.strict_precision = strict_precision

        self.qgv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qgv = self.qgv(x)
        q, gate_raw, value_raw = qgv.split(C, dim=-1)

        q = q.reshape(B, T, self.n_head, self.head_dim)
        gate_raw = gate_raw.reshape(B, T, self.n_head, self.head_dim)
        value_raw = value_raw.reshape(B, T, self.n_head, self.head_dim)

        gate = torch.sigmoid(gate_raw)
        A = self.gate_min + (self.gate_max - self.gate_min) * gate
        V = torch.tanh(value_raw)
        B_write = (1.0 - A) * V

        scan_dtype = resolve_scan_dtype(
            self.precision,
            x.device,
            strict_precision=self.strict_precision,
        )
        E = pulse_scan(A.to(scan_dtype).contiguous(), B_write.to(scan_dtype).contiguous())

        E = E.reshape(B, T, C)
        q = q.reshape(B, T, C).to(E.dtype)

        E = E * torch.rsqrt(E.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        y = torch.sigmoid(q) * E
        y = y.to(x.dtype)
        y = self.out_proj(y)
        y = self.dropout(y)
        return y
