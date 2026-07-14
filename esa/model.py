# Copyright 2026 Zameer Hussain and Akhtar Hussain
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .generation import (
    GenerationResult,
    GenerationStats,
    sample_next_token,
    parse_engine_spec,
)
from .layer import ESA


@dataclass
class ESAModelConfig:
    vocab_size: int
    block: int = 512
    n_layer: int = 6
    head: int = 6
    embd: int = 384
    dropout: float = 0.1
    bias: bool = True
    backend: str = "thunder"
    precision: str = "fp16"
    compass: int | None = None
    training_compile: bool = True
    training_compile_mode: str = "reduce-overhead"
    training_compile_fullgraph: bool = False
    gate_min: float = 0.80
    gate_max: float = 0.995
    eps: float = 1e-5
    tie_embeddings: bool = True
    format_version: int = 1

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "ESAModelConfig":
        allowed = cls.__dataclass_fields__.keys()
        return cls(
            **{
                key: value
                for key, value in data.items()
                if key in allowed
            }
        )


class _LayerNorm(nn.Module):
    def __init__(
        self,
        ndim: int,
        bias: bool,
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.ones(ndim)
        )

        self.bias = (
            nn.Parameter(torch.zeros(ndim))
            if bias
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.weight.shape,
            self.weight,
            self.bias,
            1e-5,
        )


class _MLP(nn.Module):
    def __init__(
        self,
        cfg: ESAModelConfig,
    ):
        super().__init__()

        self.fc = nn.Linear(
            cfg.embd,
            4 * cfg.embd,
            bias=cfg.bias,
        )

        self.proj = nn.Linear(
            4 * cfg.embd,
            cfg.embd,
            bias=cfg.bias,
        )

        self.drop = nn.Dropout(
            cfg.dropout
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.drop(
            self.proj(
                F.gelu(
                    self.fc(x)
                )
            )
        )


class _ESABlock(nn.Module):
    def __init__(
        self,
        cfg: ESAModelConfig,
    ):
        super().__init__()

        self.ln1 = _LayerNorm(
            cfg.embd,
            cfg.bias,
        )

        self.esa = ESA(
            embd=cfg.embd,
            head=cfg.head,
            block=cfg.block,
            backend=cfg.backend,
            precision=cfg.precision,
            compass=cfg.compass,
            dropout=cfg.dropout,
            gate_min=cfg.gate_min,
            gate_max=cfg.gate_max,
            eps=cfg.eps,
            device=None,
        )

        self.ln2 = _LayerNorm(
            cfg.embd,
            cfg.bias,
        )

        self.mlp = _MLP(cfg)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.esa(
            self.ln1(x)
        )

        x = x + self.mlp(
            self.ln2(x)
        )

        return x

    @torch.no_grad()
    def prefill(
        self,
        x: torch.Tensor,
        *,
        backend: str | None = None,
        compass: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y, state = self.esa.prefill(
            self.ln1(x),
            backend=backend,
            compass=compass,
        )
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x, state

    # Backward-compatible alias.
    lightning_prefill = prefill

    def lightning_step(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y, new_state = self.esa.decode_step(
            self.ln1(x),
            state,
        )

        x = x + y
        x = x + self.mlp(
            self.ln2(x)
        )

        return x, new_state


class ESAModel(nn.Module):
    """
    Complete causal language model built from ESA layers.

    Public lifecycle:
        model(...)
        model.generate(...)
        model.save(...)
        ESAModel.load(...)
    """

    def __init__(
        self,
        config: ESAModelConfig | None = None,
        **kwargs: Any,
    ):
        super().__init__()

        if config is None:
            config = ESAModelConfig(
                **kwargs
            )
        elif kwargs:
            raise TypeError(
                "Pass either config=... or keyword configuration, not both."
            )

        self.config = config

        self.wte = nn.Embedding(
            config.vocab_size,
            config.embd,
        )

        self.wpe = nn.Embedding(
            config.block,
            config.embd,
        )

        self.drop = nn.Dropout(
            config.dropout
        )

        self.blocks = nn.ModuleList(
            [
                _ESABlock(config)
                for _ in range(config.n_layer)
            ]
        )

        self.ln_f = _LayerNorm(
            config.embd,
            config.bias,
        )

        self.lm_head = nn.Linear(
            config.embd,
            config.vocab_size,
            bias=False,
        )

        if config.tie_embeddings:
            self.wte.weight = self.lm_head.weight

        self.apply(
            self._init_weights
        )

        for name, parameter in self.named_parameters():
            if (
                name.endswith(
                    (
                        "proj.weight",
                        "out_proj.weight",
                    )
                )
                and parameter.ndim >= 2
            ):
                nn.init.normal_(
                    parameter,
                    mean=0.0,
                    std=0.02
                    / math.sqrt(
                        2 * config.n_layer
                    ),
                )

        self._compiled_lightning_step = None

        self._compiled_lightning_key = None
        self._compiled_prefill_cache: dict[tuple[Any, ...], Any] = {}
        self._prefill_compile_failures: set[tuple[Any, ...]] = set()
        self._compiled_training_forward = None
        self._compiled_training_key = None
        self._training_compile_failed = False
        self._compile_warnings_emitted: set[str] = set()
    def _init_weights(
        self,
        module: nn.Module,
    ) -> None:
        if isinstance(
            module,
            nn.Linear,
        ):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

            if module.bias is not None:
                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(
            module,
            nn.Embedding,
        ):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    @property
    def device(
        self,
    ) -> torch.device:
        return self.wte.weight.device

    def _forward_eager(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, T = input_ids.shape
        if T > self.config.block:
            raise ValueError(
                f"Sequence length {T} exceeds block {self.config.block}."
            )
        pos = torch.arange(
            T,
            dtype=torch.long,
            device=input_ids.device,
        )
        x = self.drop(self.wte(input_ids) + self.wpe(pos)[None, :, :])
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
        return logits, loss

    def compile_training(
        self,
        *,
        mode: str | None = None,
        fullgraph: bool | None = None,
    ) -> "ESAModel":
        """Compile the full training forward path and cache it."""
        if not hasattr(torch, "compile"):
            return self
        mode = mode or self.config.training_compile_mode
        fullgraph = (
            self.config.training_compile_fullgraph
            if fullgraph is None
            else bool(fullgraph)
        )
        key = (mode, fullgraph)
        if self._compiled_training_key == key and self._compiled_training_forward is not None:
            return self
        try:
            self._compiled_training_forward = torch.compile(
                self._forward_eager,
                mode=mode,
                fullgraph=fullgraph,
            )
            self._compiled_training_key = key
            self._training_compile_failed = False
        except Exception as exc:
            self._compiled_training_forward = None
            self._training_compile_failed = True
            self._warn_compile_fallback("training", exc)
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        use_compiled_training = (
            bool(self.config.training_compile)
            and self.device.type == "cuda"
            and self.training
            and targets is not None
            and not self._training_compile_failed
        )
        if use_compiled_training:
            if self._compiled_training_forward is None:
                self.compile_training()
            if self._compiled_training_forward is not None:
                return self._compiled_training_forward(input_ids, targets)
        return self._forward_eager(input_ids, targets)

    @torch.no_grad()
    def _prefill_eager(
        self,
        input_ids: torch.Tensor,
        *,
        backend: str | None = None,
        compass: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must be [B,T], "
                f"got {tuple(input_ids.shape)}"
            )
        if input_ids.size(1) > self.config.block:
            input_ids = input_ids[:, -self.config.block:]
        T = input_ids.size(1)
        if T <= 0:
            raise ValueError("Prefill requires at least one token.")
        pos = torch.arange(T, device=input_ids.device)
        x = self.drop(self.wte(input_ids) + self.wpe(pos)[None, :, :])
        states = []
        for block in self.blocks:
            x, state = block.prefill(
                x,
                backend=backend,
                compass=compass,
            )
            states.append(state)
        states_out = torch.stack(states, dim=0).contiguous()
        logits = self.lm_head(self.ln_f(x[:, -1]))
        return logits, states_out, T

    def _warn_compile_fallback(self, component: str, exc: Exception) -> None:
        import warnings

        if component in self._compile_warnings_emitted:
            return
        self._compile_warnings_emitted.add(component)
        warnings.warn(
            f"ESA {component} compilation failed; falling back to eager execution: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )

    def _cudagraph_mark_step_begin(self) -> None:
        """Mark a new CUDA-graph iteration when the PyTorch API is available.

        ``torch.compile(mode="reduce-overhead")`` may use CUDA graphs. ESA
        Lightning carries recurrent state from one compiled invocation into the
        next, so explicitly marking decode-step boundaries prevents PyTorch from
        treating successive autoregressive steps as one graph iteration.
        """
        if self.device.type != "cuda":
            return

        compiler = getattr(torch, "compiler", None)
        marker = getattr(compiler, "cudagraph_mark_step_begin", None)

        if marker is not None:
            marker()

    @torch.no_grad()
    def prefill(
        self,
        input_ids: torch.Tensor,
        *,
        engine: str = "thunder_compiled_16",
        compile_mode: str = "reduce-overhead",
        fullgraph: bool = False,
        dynamic: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Run prompt prefill with a selectable ESA execution engine."""
        spec = parse_engine_spec(engine)
        backend = spec.backend
        compass = spec.compass

        def eager(ids: torch.Tensor):
            return self._prefill_eager(
                ids,
                backend=backend,
                compass=compass,
            )

        if spec.compiled and self.device.type == "cuda" and hasattr(torch, "compile"):
            key = (
                spec.backend,
                spec.compass,
                compile_mode,
                bool(fullgraph),
                bool(dynamic),
            )
            compiled_fn = self._compiled_prefill_cache.get(key)
            if compiled_fn is None and key not in self._prefill_compile_failures:
                try:
                    compiled_fn = torch.compile(
                        eager,
                        mode=compile_mode,
                        fullgraph=fullgraph,
                        dynamic=dynamic,
                    )
                    self._compiled_prefill_cache[key] = compiled_fn
                except Exception as exc:
                    self._prefill_compile_failures.add(key)
                    self._warn_compile_fallback("prefill", exc)
            if compiled_fn is not None:
                try:
                    return compiled_fn(input_ids)
                except Exception as exc:
                    self._prefill_compile_failures.add(key)
                    self._compiled_prefill_cache.pop(key, None)
                    self._warn_compile_fallback("prefill", exc)

        return eager(input_ids)

    @torch.no_grad()
    def lightning_prefill(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Backward-compatible v2.1 prefill using the model's configured backend."""
        return self._prefill_eager(input_ids)

    def lightning_step(
        self,
        token: torch.Tensor,
        states: torch.Tensor,
        pos_tensor: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        x = (
            self.wte(token)
            + self.wpe(pos_tensor)[None, :]
        )

        x = self.drop(x)

        new_states = []

        for index, block in enumerate(
            self.blocks
        ):
            x, state_i = block.lightning_step(
                x,
                states[index],
            )

            new_states.append(
                state_i
            )

        states_out = torch.stack(
            new_states,
            dim=0,
        ).contiguous()

        logits = self.lm_head(
            self.ln_f(x)
        )

        return logits, states_out

    def compile_generation(
        self,
        *,
        mode: str = "reduce-overhead",
        fullgraph: bool = False,
    ) -> "ESAModel":
        if not hasattr(torch, "compile") or self.device.type != "cuda":
            return self

        key = (mode, bool(fullgraph))
        if (
            self._compiled_lightning_step is not None
            and self._compiled_lightning_key == key
        ):
            return self

        try:
            self._compiled_lightning_step = torch.compile(
                self.lightning_step,
                mode=mode,
                fullgraph=fullgraph,
                dynamic=False,
            )
            self._compiled_lightning_key = key
        except Exception as exc:
            self._compiled_lightning_step = None
            self._compiled_lightning_key = None
            self._warn_compile_fallback("runtime", exc)
        return self

    @torch.inference_mode()
    def generate(
        self,
        prompt: str | torch.Tensor | None = None,
        *,
        tokenizer: Any | None = None,
        input_ids: torch.Tensor | None = None,
        seek: int = 128,
        prefill: str = "thunder_compiled_16",
        
        runtime: str = "lightning",
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_token_id: int | None = None,
        seed: int | None = None,
        compile: bool = True,
        compile_mode: str = "reduce-overhead",
        progress_interval: int | None = None,
        stats: bool = False,
        max_new_tokens: int | None = None,
    ) -> torch.Tensor | str | GenerationResult:
        """Generate text with optimized ESA defaults.

        Normal users can pass raw text positionally. ``input_ids`` and
        ``max_new_tokens`` remain available for backward compatibility.
        """
        if max_new_tokens is not None:
            if seek != 128 and int(seek) != int(max_new_tokens):
                raise ValueError(
                    "Pass either seek or max_new_tokens, not conflicting values for both."
                )
            seek = int(max_new_tokens)
        seek = int(seek)
        if seek <= 0:
            raise ValueError("seek must be positive.")

        # Backward compatibility: model.generate(tensor, ...).
        if torch.is_tensor(prompt):
            if input_ids is not None:
                raise ValueError("Provide token IDs either positionally or via input_ids, not both.")
            input_ids = prompt
            prompt = None

        if input_ids is None:
            if prompt is None or tokenizer is None:
                raise ValueError(
                    "Provide a text prompt with tokenizer, or use the advanced input_ids API."
                )
            if hasattr(tokenizer, "encode_ordinary"):
                ids = tokenizer.encode_ordinary(prompt)
            elif hasattr(tokenizer, "encode"):
                ids = tokenizer.encode(prompt)
            else:
                raise TypeError(
                    "tokenizer must expose encode_ordinary() or encode()."
                )
            input_ids = torch.tensor(
                ids,
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(0)
        else:
            input_ids = input_ids.to(self.device)

        runtime_spec = parse_engine_spec(runtime)
        if runtime_spec.backend != "lightning":
            raise ValueError(
                "Autoregressive decode currently supports runtime='lightning' only. "
                "Thunder/Flare/Pulse are selectable prefill engines."
            )
        compile_runtime = bool(compile or runtime_spec.compiled)

        was_training = self.training
        self.eval()
        try:
            if seed is not None:
                torch.manual_seed(int(seed))
                if self.device.type == "cuda":
                    torch.cuda.manual_seed_all(int(seed))

            def sync() -> None:
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)

            prompt_tokens = int(input_ids.size(1))
            sync()
            total_start = time.perf_counter()
            prefill_start = total_start
            logits, states, prefill_len = self.prefill(
                input_ids,
                engine=prefill,
                compile_mode=compile_mode,
                fullgraph=False,
                dynamic=True,
            )

            # A compiled prefill may return CUDA-graph-managed output storage.
            # Lightning decode carries ESA state across many later invocations,
            # so move the recurrent state into stable, independently owned
            # storage before the first decode step.
            if states.is_cuda:
                states = states.clone()

            sync()
            prefill_seconds = time.perf_counter() - prefill_start

            next_token = sample_next_token(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            generated = [next_token]

            step_fn = self.lightning_step
            using_compiled_runtime = False

            if compile_runtime and self.device.type == "cuda":
                key = (compile_mode, False)

                if (
                    self._compiled_lightning_step is None
                    or self._compiled_lightning_key != key
                ):
                    self.compile_generation(
                        mode=compile_mode,
                        fullgraph=False,
                    )

                if self._compiled_lightning_step is not None:
                    step_fn = self._compiled_lightning_step
                    using_compiled_runtime = True

            decode_target = seek - 1
            sync()
            decode_start = time.perf_counter()
            for step in range(decode_target):
                position = (prefill_len + step) % self.config.block
                pos_tensor = torch.tensor(
                    position,
                    device=self.device,
                    dtype=torch.long,
                )
                # ``reduce-overhead`` may capture the Lightning step in a
                # CUDA graph. Each autoregressive token is a new logical graph
                # iteration, and the recurrent state must survive subsequent
                # graph replays.
                if using_compiled_runtime:
                    self._cudagraph_mark_step_begin()

                logits, states_out = step_fn(
                    next_token.squeeze(1),
                    states,
                    pos_tensor,
                )

                # Never carry CUDA-graph-owned output storage directly into the
                # next token step. ESA state is tiny, so this correctness copy
                # is inexpensive compared with a KV cache.
                states = (
                    states_out.clone()
                    if states_out.is_cuda
                    else states_out
                )
                next_token = sample_next_token(
                    logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                generated.append(next_token)

                if (
                    eos_token_id is not None
                    and bool((next_token == int(eos_token_id)).all())
                ):
                    break

                if progress_interval and (step + 1) % int(progress_interval) == 0:
                    sync()
                    elapsed = time.perf_counter() - decode_start
                    done = step + 1
                    print(
                        f"ESA-Lightning {done:,}/{decode_target:,} | "
                        f"{done/max(elapsed, 1e-9):,.2f} tok/s"
                    )

            sync()
            decode_seconds = time.perf_counter() - decode_start
            total_seconds = time.perf_counter() - total_start
            generated_ids = torch.cat(generated, dim=1)
            sequences = torch.cat([input_ids, generated_ids], dim=1)
            state_bytes = int(states.numel() * states.element_size())
            generation_stats = GenerationStats(
                prompt_tokens=prompt_tokens,
                prefill_tokens=int(prefill_len),
                generated_tokens=int(generated_ids.size(1)),
                decode_steps=max(0, int(generated_ids.size(1)) - 1),
                prefill_seconds=prefill_seconds,
                decode_seconds=decode_seconds,
                decode_tok_s=max(0, int(generated_ids.size(1)) - 1)
                / max(decode_seconds, 1e-9),
                total_seconds=total_seconds,
                state_bytes=state_bytes,
                state_mb=state_bytes / 1024**2,
            )
            result = GenerationResult(
                sequences=sequences,
                generated_ids=generated_ids,
                stats=generation_stats,
            )
            if tokenizer is not None:
                if sequences.size(0) == 1:
                    result.text = tokenizer.decode(
                        sequences[0].detach().cpu().tolist()
                    )
                else:
                    result.text = [
                        tokenizer.decode(row.detach().cpu().tolist())
                        for row in sequences
                    ]
            if stats:
                return result
            if result.text is not None:
                return result.text
            return sequences
        finally:
            self.train(was_training)

    @torch.inference_mode()
    def generate_ids(
        self,
        input_ids: torch.Tensor,
        *,
        seek: int = 128,
        **kwargs: Any,
    ) -> torch.Tensor | GenerationResult:
        """Advanced token-level generation API."""
        return self.generate(
            input_ids=input_ids,
            seek=seek,
            **kwargs,
        )

    def model_info(
        self,
    ) -> dict[str, Any]:
        return {
            **asdict(
                self.config
            ),
            "parameters": sum(
                parameter.numel()
                for parameter
                in self.parameters()
            ),
            "device": str(
                self.device
            ),
            "generation_engine": (
                "ESA-Lightning"
            ),
        }

    def save(
        self,
        path: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        path = Path(path)
        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            path
            / "config.json"
        ).write_text(
            json.dumps(
                asdict(
                    self.config
                ),
                indent=2,
            ),
            encoding="utf-8",
        )

        torch.save(
            self.state_dict(),
            path / "model.pt",
        )

        meta = {
            "format_version": (
                self.config.format_version
            ),
            "architecture": "ESAModel",
            "generation_engine": (
                "ESA-Lightning"
            ),
            "backend": (
                self.config.backend
            ),
        }

        if metadata:
            meta.update(
                metadata
            )

        (
            path
            / "metadata.json"
        ).write_text(
            json.dumps(
                meta,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        return path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
        strict: bool = True,
    ) -> "ESAModel":
        path = Path(path)

        config = ESAModelConfig.from_dict(
            json.loads(
                (
                    path
                    / "config.json"
                ).read_text(
                    encoding="utf-8"
                )
            )
        )

        model = cls(
            config
        )

        state = torch.load(
            path / "model.pt",
            map_location=device,
            weights_only=True,
        )

        model.load_state_dict(
            state,
            strict=strict,
        )

        return model.to(
            device
        )