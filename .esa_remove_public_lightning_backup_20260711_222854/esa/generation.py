# Copyright 2026 Zameer Hussain and Akhtar Hussain
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .precision import resolve_scan_dtype


@dataclass(frozen=True)
class GenerationStats:
    prompt_tokens: int
    prefill_tokens: int
    generated_tokens: int
    decode_steps: int
    prefill_seconds: float
    decode_seconds: float
    decode_tok_s: float
    total_seconds: float
    state_bytes: int
    state_mb: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationResult:
    sequences: torch.Tensor
    generated_ids: torch.Tensor
    stats: GenerationStats
    text: str | list[str] | None = None


def _unwrap_backend(module: torch.nn.Module) -> torch.nn.Module:
    current = getattr(module, "layer", module)
    seen: set[int] = set()

    while hasattr(current, "_orig_mod") and id(current) not in seen:
        seen.add(id(current))
        current = current._orig_mod

    return current


def _backend_name(module: torch.nn.Module) -> str:
    if hasattr(module, "backend"):
        return str(module.backend).lower()

    name = _unwrap_backend(module).__class__.__name__.lower()

    for candidate in (
        "flare",
        "thunder",
        "pulse",
        "lightning",
    ):
        if candidate in name:
            return candidate

    return name


def _dimensions(
    module: torch.nn.Module,
) -> tuple[int, int, int]:
    backend = _unwrap_backend(module)

    embd = getattr(backend, "embd", None)
    head = getattr(backend, "head", None)
    head_dim = getattr(backend, "head_dim", None)

    if embd is None or head is None:
        raise AttributeError(
            "ESA backend must expose embd and head."
        )

    embd = int(embd)
    head = int(head)

    if head_dim is None:
        head_dim = embd // head

    return embd, head, int(head_dim)


def _state_dtype(
    module: torch.nn.Module,
    device: torch.device,
    input_dtype: torch.dtype,
) -> torch.dtype:
    backend = _unwrap_backend(module)
    name = _backend_name(module)

    if name == "flare":
        return input_dtype

    precision = str(
        getattr(
            backend,
            "precision",
            getattr(module, "precision", "fp16"),
        )
    )

    return resolve_scan_dtype(
        precision,
        device,
        strict_precision=bool(
            getattr(
                backend,
                "strict_precision",
                False,
            )
        ),
    )


def _project_affine_terms(
    module: torch.nn.Module,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    backend = _unwrap_backend(module)
    embd, head, head_dim = _dimensions(module)

    if x.ndim != 3 or x.size(-1) != embd:
        raise ValueError(
            f"Expected x [B,T,{embd}], got {tuple(x.shape)}"
        )

    B, T, C = x.shape

    q, gate_raw, value_raw = backend.qgv(x).split(
        C,
        dim=-1,
    )

    q = q.reshape(B, T, head, head_dim)
    gate_raw = gate_raw.reshape(
        B,
        T,
        head,
        head_dim,
    )
    value_raw = value_raw.reshape(
        B,
        T,
        head,
        head_dim,
    )

    gate = torch.sigmoid(gate_raw)
    A = backend.gate_min + (
        backend.gate_max - backend.gate_min
    ) * gate

    V = torch.tanh(value_raw)
    B_write = (1.0 - A) * V

    return q, A, B_write


def _backend_scan(
    module: torch.nn.Module,
    A: torch.Tensor,
    B_write: torch.Tensor,
) -> torch.Tensor:
    backend = _unwrap_backend(module)
    name = _backend_name(module)

    if name == "flare":
        from .backends.flare import flare_scan

        return flare_scan(
            A.contiguous(),
            B_write.contiguous(),
            block_ch=int(
                getattr(
                    backend,
                    "block_ch",
                    128,
                )
            ),
        )

    dtype = _state_dtype(
        module,
        A.device,
        A.dtype,
    )

    A = A.to(dtype).contiguous()
    B_write = B_write.to(dtype).contiguous()

    if name == "thunder":
        from .backends.thunder import thunder_scan

        return thunder_scan(
            A,
            B_write,
            compass=int(
                getattr(
                    backend,
                    "compass",
                    16,
                )
            ),
        )

    if name == "pulse":
        from .backends.pulse import pulse_scan
        return pulse_scan(A, B_write)

    if name == "lightning":
        from .backends.lightning import lightning_scan

        return lightning_scan(
            A,
            B_write,
            compass=int(
                getattr(
                    backend,
                    "compass",
                    4,
                )
            ),
        )

    raise ValueError(
        f"Unsupported ESA backend for generation: {name!r}"
    )


def _readout(
    module: torch.nn.Module,
    q: torch.Tensor,
    states: torch.Tensor,
    *,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    backend = _unwrap_backend(module)

    B, T, H, D = states.shape
    C = H * D

    E = states.reshape(B, T, C)
    q = q.reshape(B, T, C).to(E.dtype)

    E = E * torch.rsqrt(
        E.pow(2).mean(dim=-1, keepdim=True)
        + float(backend.eps)
    )

    y = torch.sigmoid(q) * E
    y = backend.out_proj(y.to(output_dtype))

    return backend.dropout(y)


def lightning_init_state(
    module: torch.nn.Module,
    batch: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    layout: str = "heads",
) -> torch.Tensor:
    backend = _unwrap_backend(module)
    embd, head, head_dim = _dimensions(module)

    if device is None:
        device = backend.qgv.weight.device

    device = torch.device(device)

    if dtype is None:
        dtype = _state_dtype(
            module,
            device,
            backend.qgv.weight.dtype,
        )

    if layout == "heads":
        shape = (batch, head, head_dim)
    elif layout == "flat":
        shape = (batch, embd)
    else:
        raise ValueError(
            "layout must be 'heads' or 'flat'"
        )

    return torch.zeros(
        shape,
        device=device,
        dtype=dtype,
    )


@torch.no_grad()
def lightning_prefill(
    module: torch.nn.Module,
    x: torch.Tensor,
    state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    ESA-Lightning prompt prefill.

    Exact recurrence:
        state_t = A_t * state_{t-1} + B_write_t
    """
    if x.ndim != 3 or x.size(1) <= 0:
        raise ValueError(
            f"Prefill expects non-empty [B,T,C], got {tuple(x.shape)}"
        )

    B = x.size(0)
    _, head, head_dim = _dimensions(module)

    q, A, B_write = _project_affine_terms(
        module,
        x,
    )

    if state is None:
        states = _backend_scan(
            module,
            A,
            B_write,
        )
    else:
        current = (
            state.reshape(B, head, head_dim)
            if state.ndim == 2
            else state
        )

        outputs = []

        for t in range(x.size(1)):
            current = (
                A[:, t].to(current.dtype) * current
                + B_write[:, t].to(current.dtype)
            )
            outputs.append(current)

        states = torch.stack(outputs, dim=1)

    y = _readout(
        module,
        q,
        states,
        output_dtype=x.dtype,
    )

    return y, states[:, -1].contiguous()


def lightning_decode_step(
    module: torch.nn.Module,
    x: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Exact one-token ESA-Lightning decode step.

    No token history is replayed.
    """
    if x.ndim == 2:
        x3 = x.unsqueeze(1)
        squeeze = True
    elif x.ndim == 3 and x.size(1) == 1:
        x3 = x
        squeeze = False
    else:
        raise ValueError(
            "decode_step expects [B,C] or [B,1,C], "
            f"got {tuple(x.shape)}"
        )

    B = x3.size(0)
    _, head, head_dim = _dimensions(module)

    flat_state = state.ndim == 2

    state_h = (
        state.reshape(B, head, head_dim)
        if flat_state
        else state
    )

    q, A, B_write = _project_affine_terms(
        module,
        x3,
    )

    new_state_h = (
        A[:, 0].to(state_h.dtype) * state_h
        + B_write[:, 0].to(state_h.dtype)
    )

    y = _readout(
        module,
        q,
        new_state_h.unsqueeze(1),
        output_dtype=x3.dtype,
    )

    if squeeze:
        y = y[:, 0]

    new_state = (
        new_state_h.reshape(B, -1).contiguous()
        if flat_state
        else new_state_h.contiguous()
    )

    return y, new_state


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    if logits.ndim == 3:
        logits = logits[:, -1]

    if logits.ndim != 2:
        raise ValueError(
            f"Expected logits [B,V] or [B,T,V], got {tuple(logits.shape)}"
        )

    if temperature <= 0:
        return torch.argmax(
            logits,
            dim=-1,
            keepdim=True,
        )

    logits = logits / max(
        float(temperature),
        1e-5,
    )

    if top_k is not None:
        k = min(
            int(top_k),
            logits.size(-1),
        )

        values, _ = torch.topk(
            logits,
            k,
        )

        logits = logits.masked_fill(
            logits < values[:, [-1]],
            float("-inf"),
        )

    if (
        top_p is not None
        and 0.0 < float(top_p) < 1.0
    ):
        sorted_logits, sorted_indices = torch.sort(
            logits,
            descending=True,
        )

        sorted_probs = F.softmax(
            sorted_logits,
            dim=-1,
        )

        cumulative = torch.cumsum(
            sorted_probs,
            dim=-1,
        )

        remove = cumulative > float(top_p)
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False

        sorted_logits = sorted_logits.masked_fill(
            remove,
            float("-inf"),
        )

        logits = torch.full_like(
            logits,
            float("-inf"),
        )

        logits.scatter_(
            1,
            sorted_indices,
            sorted_logits,
        )

    return torch.multinomial(
        F.softmax(
            logits,
            dim=-1,
        ),
        num_samples=1,
    )
