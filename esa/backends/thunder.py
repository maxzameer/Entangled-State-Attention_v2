# Copyright 2026 Zameer Hussain and Akhtar Hussain
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..precision import resolve_scan_dtype


def associative_chunk_scan(
    A_chunk: torch.Tensor,
    B_chunk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inclusive affine scan over chunk summaries.

    Composition rule for:

        E_t = A_t * E_{t-1} + B_t

    is:

        (A2, B2) ∘ (A1, B1) = (A2 * A1, A2 * B1 + B2)

    Args:
        A_chunk: Tensor of shape [B, G, H, D].
        B_chunk: Tensor of shape [B, G, H, D].

    Returns:
        Tuple (A_scan, B_scan), both of shape [B, G, H, D].
    """
    if A_chunk.shape != B_chunk.shape:
        raise ValueError(
            f"A_chunk and B_chunk must have same shape, "
            f"got {A_chunk.shape} and {B_chunk.shape}"
        )

    if A_chunk.dim() != 4:
        raise ValueError(
            f"expected A_chunk/B_chunk shape [B,G,H,D], got {A_chunk.shape}"
        )

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
        B_next[:, step:] = (
            A_prev[:, step:] * B_prev[:, :-step] + B_prev[:, step:]
        )

        A = A_next
        B = B_next
        step *= 2

    return A, B


def thunder_scan(
    A: torch.Tensor,
    B_write: torch.Tensor,
    compass: int = 16,
    c: int | None = None,
) -> torch.Tensor:
    """Thunder chunked ESA scan.

    Computes the recurrence:

        E_t = A_t * E_{t-1} + B_t

    using local chunk scans followed by an associative scan over chunk
    summaries.

    Args:
        A: Retention tensor of shape [B, T, H, D].
        B_write: Write tensor of shape [B, T, H, D].
        compass: Internal ESA Thunder chunk size. Default is 16.
        c: Old compatibility name for compass.

    Returns:
        ESA state tensor E of shape [B, T, H, D].
    """
    if c is not None:
        compass = c

    if A.shape != B_write.shape:
        raise ValueError(
            f"A and B_write must have same shape, got {A.shape} and {B_write.shape}"
        )

    if A.dim() != 4:
        raise ValueError(f"expected A/B_write shape [B,T,H,D], got {A.shape}")

    if not isinstance(compass, int) or compass <= 0:
        raise ValueError(
            f"compass must be a positive integer, got {compass!r}"
        )

    Bsz, T, H, D = A.shape

    pad = (-T) % compass

    if pad > 0:
        A = F.pad(A, (0, 0, 0, 0, 0, pad), value=1.0)
        B_write = F.pad(B_write, (0, 0, 0, 0, 0, pad), value=0.0)

    Tp = A.size(1)
    G = Tp // compass

    A5 = A.reshape(Bsz, G, compass, H, D)
    B5 = B_write.reshape(Bsz, G, compass, H, D)

    state = B_write.new_zeros(Bsz, G, H, D)
    transition = A.new_ones(Bsz, G, H, D)

    local_states = []
    prefix_As = []

    # Local recurrence inside each compass chunk.
    # compass is intentionally small; optimized default is 16.
    for i in range(compass):
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

    Thunder is the default ESA backend. It uses the internal ESA
    compass value for chunked associative state scanning.

    Public users normally should not pass compass, gate_min, gate_max,
    or eps. These are ESA algorithm defaults.

    Clean public usage:

        ESA(embd=128, head=4, batch=16, block=1024, backend="thunder")

    Backward-compatible usage is also supported:

        ESA(n_embd=128, n_head=4, backend="thunder")

    This implementation follows the CF-ESA-c16-FP16Scan path with:

        compass=16
        precision="fp16"
        gate_min=0.80
        gate_max=0.995
        eps=1e-5
    """

    def __init__(
        self,

        # New clean names
        embd: int | None = None,
        head: int = 4,

        # Old names for compatibility
        n_embd: int | None = None,
        n_head: int | None = None,

        dropout: float = 0.0,

        # New internal name
        compass: int = 16,

        # Old name for compatibility
        c: int | None = None,

        precision: str = "fp16",
        gate_min: float = 0.80,
        gate_max: float = 0.995,
        eps: float = 1e-5,
        strict_precision: bool = False,
    ):
        super().__init__()

        # ----------------------------------------------------
        # Resolve new names and old names
        # ----------------------------------------------------
        if embd is None:
            embd = n_embd

        if n_head is not None:
            head = n_head

        if c is not None:
            compass = c

        if embd is None:
            raise ValueError(
                "ThunderESA requires embd. Example: ThunderESA(embd=128, head=4)"
            )

        if embd % head != 0:
            raise ValueError(
                f"embd must be divisible by head, got embd={embd}, head={head}"
            )

        if not isinstance(compass, int) or compass <= 0:
            raise ValueError(
                f"compass must be a positive integer, got {compass!r}"
            )

        # Clean public/internal attributes
        self.embd = embd
        self.head = head
        self.head_dim = embd // head
        self.compass = compass

        # Backward-compatible attributes
        self.n_embd = embd
        self.n_head = head
        self.c = compass

        self.precision = precision
        self.gate_min = gate_min
        self.gate_max = gate_max
        self.eps = eps
        self.strict_precision = strict_precision

        # Match the optimized benchmark: bias=False.
        self.qgv = nn.Linear(embd, 3 * embd, bias=False)
        self.out_proj = nn.Linear(embd, embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected input shape [B,T,C], got {x.shape}")

        B, T, C = x.shape

        if C != self.embd:
            raise ValueError(
                f"expected embedding dim {self.embd}, got input dim {C}"
            )

        qgv = self.qgv(x)
        q, gate_raw, value_raw = qgv.split(C, dim=-1)

        q = q.reshape(B, T, self.head, self.head_dim)
        gate_raw = gate_raw.reshape(B, T, self.head, self.head_dim)
        value_raw = value_raw.reshape(B, T, self.head, self.head_dim)

        gate = torch.sigmoid(gate_raw)
        A = self.gate_min + (self.gate_max - self.gate_min) * gate

        V = torch.tanh(value_raw)
        B_write = (1.0 - A) * V

        scan_dtype = resolve_scan_dtype(
            self.precision,
            x.device,
            strict_precision=self.strict_precision,
        )

        # Match CF-ESA-c16-FP16Scan when precision="fp16".
        A_scan = A.to(scan_dtype).contiguous()
        B_scan = B_write.to(scan_dtype).contiguous()

        E = thunder_scan(
            A_scan,
            B_scan,
            compass=self.compass,
        )

        E = E.reshape(B, T, C)
        q = q.reshape(B, T, C).to(E.dtype)

        # Benchmark-matching normalization epsilon: 1e-5.
        E = E * torch.rsqrt(E.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        y = torch.sigmoid(q) * E
        y = y.to(x.dtype)

        y = self.out_proj(y)
        y = self.dropout(y)

        return y