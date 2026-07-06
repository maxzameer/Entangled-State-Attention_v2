# Copyright 2026 Zameer Hussain and Akhtar Hussain
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..precision import resolve_scan_dtype


def associative_chunk_scan(A_chunk: torch.Tensor, B_chunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Inclusive affine scan over chunk summaries.

    Composition rule for ``E_t = A_t * E_{t-1} + B_t``:

    ``(A2, B2) ∘ (A1, B1) = (A2 * A1, A2 * B1 + B2)``.
    """
    A = A_chunk
    B = B_chunk
    G = A.size(1)
    step = 1

    while step < G:
        A_prev = A
        B_prev = B

        A_next = A.clone()
        B_next = B.clone()

        A_next[:, step:] = A_prev[:, step:] * A_prev[:, :-step]
        B_next[:, step:] = A_prev[:, step:] * B_prev[:, :-step] + B_prev[:, step:]

        A = A_next
        B = B_next
        step *= 2

    return A, B


def thunder_scan(A: torch.Tensor, B_write: torch.Tensor, c: int = 16) -> torch.Tensor:
    """Thunder chunked ESA scan.

    Parameters
    ----------
    A:
        Retention tensor of shape ``[B, T, H, D]``.
    B_write:
        Write tensor of shape ``[B, T, H, D]``.
    c:
        Thunder chunked scan parameter. Typical values are 8, 16, 32, or 64.
    """
    if A.shape != B_write.shape:
        raise ValueError(f"A and B_write must have same shape, got {A.shape} and {B_write.shape}")
    if A.dim() != 4:
        raise ValueError(f"expected A/B_write shape [B,T,H,D], got {A.shape}")
    if not isinstance(c, int) or c <= 0:
        raise ValueError(f"c must be a positive integer, got {c!r}")

    Bsz, T, H, D = A.shape
    pad = (-T) % c

    if pad > 0:
        A = F.pad(A, (0, 0, 0, 0, 0, pad), value=1.0)
        B_write = F.pad(B_write, (0, 0, 0, 0, 0, pad), value=0.0)

    Tp = A.size(1)
    G = Tp // c

    A5 = A.reshape(Bsz, G, c, H, D)
    B5 = B_write.reshape(Bsz, G, c, H, D)

    state = B_write.new_zeros(Bsz, G, H, D)
    transition = A.new_ones(Bsz, G, H, D)

    local_states = []
    prefix_As = []

    # Local recurrence inside each chunk. c is intentionally small.
    for i in range(c):
        A_i = A5[:, :, i]
        B_i = B5[:, :, i]
        state = A_i * state + B_i
        transition = A_i * transition
        local_states.append(state)
        prefix_As.append(transition)

    local_state = torch.stack(local_states, dim=2)
    prefix_A = torch.stack(prefix_As, dim=2)

    A_chunk = prefix_A[:, :, -1]
    B_chunk = local_state[:, :, -1]

    _, chunk_end_state = associative_chunk_scan(A_chunk, B_chunk)

    zero = chunk_end_state.new_zeros(Bsz, 1, H, D)
    chunk_init = torch.cat([zero, chunk_end_state[:, :-1]], dim=1)

    E = prefix_A * chunk_init.unsqueeze(2) + local_state
    E = E.reshape(Bsz, Tp, H, D)

    if pad > 0:
        E = E[:, :T]

    return E


class ThunderESA(nn.Module):
    """Thunder backend: optimized chunked ESA backend.

    Thunder is the default ESA v2 backend. It supports the chunked scan
    parameter ``c`` and is intended for normal training and benchmark use.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int = 4,
        dropout: float = 0.0,
        c: int = 16,
        precision: str = "fp16",
        gate_min: float = 0.80,
        gate_max: float = 0.995,
        eps: float = 1e-6,
        strict_precision: bool = False,
    ):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if not isinstance(c, int) or c <= 0:
            raise ValueError(f"c must be a positive integer, got {c!r}")

        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.c = c
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
        A_scan = A.to(scan_dtype).contiguous()
        B_scan = B_write.to(scan_dtype).contiguous()

        E = thunder_scan(A_scan, B_scan, c=self.c)

        E = E.reshape(B, T, C)
        q = q.reshape(B, T, C).to(E.dtype)

        E = E * torch.rsqrt(E.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        y = torch.sigmoid(q) * E
        y = y.to(x.dtype)
        y = self.out_proj(y)
        y = self.dropout(y)
        return y
