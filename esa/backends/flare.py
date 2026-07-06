# Copyright 2026 Zameer Hussain and Akhtar Hussain
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import torch
import torch.nn as nn

try:  # pragma: no cover - Triton is optional and usually CUDA-only.
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
    TRITON_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover
    triton = None
    tl = None
    TRITON_AVAILABLE = False
    TRITON_ERROR = exc


if TRITON_AVAILABLE:  # pragma: no cover - requires CUDA + Triton.

    @triton.jit
    def _flare_scan_fwd_kernel(
        a_ptr,
        b_ptr,
        out_ptr,
        n_channels,
        T,
        cflat,
        BLOCK_CH: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_CH + tl.arange(0, BLOCK_CH)
        mask = offs < n_channels
        base = (offs // cflat) * (T * cflat) + (offs % cflat)
        state = tl.zeros((BLOCK_CH,), dtype=tl.float32)

        t = 0
        while t < T:
            ptr = base + t * cflat
            a_t = tl.load(a_ptr + ptr, mask=mask, other=1.0).to(tl.float32)
            b_t = tl.load(b_ptr + ptr, mask=mask, other=0.0).to(tl.float32)
            state = a_t * state + b_t
            tl.store(out_ptr + ptr, state, mask=mask)
            t += 1


    @triton.jit
    def _flare_scan_bwd_kernel(
        a_ptr,
        out_ptr,
        grad_out_ptr,
        grad_a_ptr,
        grad_b_ptr,
        n_channels,
        T,
        cflat,
        BLOCK_CH: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_CH + tl.arange(0, BLOCK_CH)
        mask = offs < n_channels
        base = (offs // cflat) * (T * cflat) + (offs % cflat)
        future = tl.zeros((BLOCK_CH,), dtype=tl.float32)

        t = T - 1
        while t >= 0:
            ptr = base + t * cflat
            grad_direct = tl.load(grad_out_ptr + ptr, mask=mask, other=0.0).to(tl.float32)
            grad_state = grad_direct + future
            prev_ptr = base + (t - 1) * cflat
            e_prev = tl.load(out_ptr + prev_ptr, mask=mask & (t > 0), other=0.0).to(tl.float32)
            a_t = tl.load(a_ptr + ptr, mask=mask, other=1.0).to(tl.float32)
            tl.store(grad_b_ptr + ptr, grad_state, mask=mask)
            tl.store(grad_a_ptr + ptr, grad_state * e_prev, mask=mask)
            future = grad_state * a_t
            t -= 1


class FlareScanFunction(torch.autograd.Function):
    """Triton-backed direct ESA scan used by the Flare backend."""

    @staticmethod
    def forward(ctx, A: torch.Tensor, B_write: torch.Tensor, block_ch: int):
        if not TRITON_AVAILABLE:
            raise RuntimeError(f"Triton is not available: {TRITON_ERROR}")
        if not A.is_cuda or not B_write.is_cuda:
            raise RuntimeError("Flare requires CUDA tensors.")
        if A.shape != B_write.shape:
            raise ValueError(f"A and B_write must have same shape, got {A.shape} and {B_write.shape}")
        if A.dim() != 4:
            raise ValueError(f"expected A/B_write shape [B,T,H,D], got {A.shape}")

        A = A.contiguous()
        B_write = B_write.contiguous()
        B, T, H, D = A.shape
        n_channels = B * H * D
        cflat = H * D
        out = torch.empty_like(B_write)

        grid = (triton.cdiv(n_channels, block_ch),)
        _flare_scan_fwd_kernel[grid](
            A,
            B_write,
            out,
            n_channels,
            T,
            cflat,
            BLOCK_CH=block_ch,
            num_warps=4,
        )

        ctx.save_for_backward(A, out)
        ctx.block_ch = block_ch
        ctx.shape_info = (B, T, H, D)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        A, out = ctx.saved_tensors
        B, T, H, D = ctx.shape_info
        grad_out = grad_out.contiguous()
        grad_A = torch.empty_like(A)
        grad_B = torch.empty_like(grad_out)
        n_channels = B * H * D
        cflat = H * D
        block_ch = ctx.block_ch

        grid = (triton.cdiv(n_channels, block_ch),)
        _flare_scan_bwd_kernel[grid](
            A,
            out,
            grad_out,
            grad_A,
            grad_B,
            n_channels,
            T,
            cflat,
            BLOCK_CH=block_ch,
            num_warps=4,
        )
        return grad_A, grad_B, None


def flare_scan(A: torch.Tensor, B_write: torch.Tensor, *, block_ch: int = 128) -> torch.Tensor:
    """Run the Flare Triton direct ESA scan."""
    return FlareScanFunction.apply(A, B_write, block_ch)


class FlareESA(nn.Module):
    """Flare backend: experimental Triton ESA backend.

    Flare does not expose Thunder's chunked scan parameter ``c``. It is a
    Triton direct-scan backend for CUDA experiments.
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
        block_ch: int = 128,
        strict_precision: bool = False,
    ):
        super().__init__()
        if not TRITON_AVAILABLE:
            raise RuntimeError(
                "backend='flare' requires Triton. Install with: "
                "pip install entangled-state-attention[triton]. "
                f"Original import error: {TRITON_ERROR}"
            )
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if precision not in {"fp16", "bf16", "fp32", "fp64", "fp8"}:
            raise ValueError(f"Unsupported precision={precision!r}")
        if precision == "fp8" and strict_precision:
            raise NotImplementedError("True FP8 Flare scan is not implemented yet.")

        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.precision = precision
        self.gate_min = gate_min
        self.gate_max = gate_max
        self.eps = eps
        self.block_ch = block_ch
        self.strict_precision = strict_precision

        self.qgv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("backend='flare' requires CUDA tensors.")
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

        # Flare kernels accumulate in fp32 internally. Inputs are contiguous and
        # usually fp16/bf16/fp32 depending on the surrounding model precision.
        E = flare_scan(A.contiguous(), B_write.contiguous(), block_ch=self.block_ch)

        E = E.reshape(B, T, C)
        q = q.reshape(B, T, C).to(E.dtype)
        E = E * torch.rsqrt(E.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        y = torch.sigmoid(q) * E
        y = y.to(x.dtype)
        y = self.out_proj(y)
        y = self.dropout(y)
        return y
