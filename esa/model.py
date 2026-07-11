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
    backend: str = "flare"
    precision: str = "fp16"
    compass: int | None = None
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
    def lightning_prefill(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y, state = self.esa.prefill(
            self.ln1(x)
        )

        x = x + y
        x = x + self.mlp(
            self.ln2(x)
        )

        return x, state

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

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
    ]:
        B, T = input_ids.shape

        if T > self.config.block:
            raise ValueError(
                f"Sequence length {T} exceeds block {self.config.block}."
            )

        pos = torch.arange(
            T,
            dtype=torch.long,
            device=input_ids.device,
        )

        x = self.drop(
            self.wte(input_ids)
            + self.wpe(pos)[None, :, :]
        )

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    logits.size(-1),
                ),
                targets.reshape(-1),
                ignore_index=-1,
            )

        return logits, loss

    @torch.no_grad()
    def lightning_prefill(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        int,
    ]:
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must be [B,T], "
                f"got {tuple(input_ids.shape)}"
            )

        if input_ids.size(1) > self.config.block:
            input_ids = input_ids[
                :,
                -self.config.block:
            ]

        T = input_ids.size(1)

        pos = torch.arange(
            T,
            device=input_ids.device,
        )

        x = self.drop(
            self.wte(input_ids)
            + self.wpe(pos)[None, :, :]
        )

        states = []

        for block in self.blocks:
            x, state = block.lightning_prefill(
                x
            )
            states.append(state)

        states = torch.stack(
            states,
            dim=0,
        ).contiguous()

        logits = self.lm_head(
            self.ln_f(
                x[:, -1]
            )
        )

        return logits, states, T

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
        self._compiled_lightning_step = torch.compile(
            self.lightning_step,
            mode=mode,
            fullgraph=fullgraph,
        )

        return self

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        prompt: str | None = None,
        tokenizer: Any | None = None,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_token_id: int | None = None,
        seed: int | None = None,
        compile: bool = True,
        compile_mode: str = "reduce-overhead",
        progress_interval: int | None = None,
        stats: bool = False,
    ) -> torch.Tensor | str | GenerationResult:
        if max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be positive."
            )

        if input_ids is None:
            if (
                prompt is None
                or tokenizer is None
            ):
                raise ValueError(
                    "Provide input_ids, or both prompt and tokenizer."
                )

            if hasattr(
                tokenizer,
                "encode_ordinary",
            ):
                ids = tokenizer.encode_ordinary(
                    prompt
                )
            elif hasattr(
                tokenizer,
                "encode",
            ):
                ids = tokenizer.encode(
                    prompt
                )
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
            input_ids = input_ids.to(
                self.device
            )

        was_training = self.training
        self.eval()

        if seed is not None:
            torch.manual_seed(
                int(seed)
            )

            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(
                    int(seed)
                )

        def sync() -> None:
            if self.device.type == "cuda":
                torch.cuda.synchronize(
                    self.device
                )

        prompt_tokens = int(
            input_ids.size(1)
        )

        sync()
        total_start = time.perf_counter()
        prefill_start = total_start

        logits, states, prefill_len = (
            self.lightning_prefill(
                input_ids
            )
        )

        sync()

        prefill_seconds = (
            time.perf_counter()
            - prefill_start
        )

        next_token = sample_next_token(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        generated = [
            next_token
        ]

        step_fn = self.lightning_step

        if compile:
            if (
                self._compiled_lightning_step
                is None
            ):
                self.compile_generation(
                    mode=compile_mode
                )

            step_fn = (
                self._compiled_lightning_step
            )

        decode_target = (
            max_new_tokens - 1
        )

        sync()
        decode_start = time.perf_counter()

        for step in range(
            decode_target
        ):
            position = (
                prefill_len + step
            ) % self.config.block

            pos_tensor = torch.tensor(
                position,
                device=self.device,
                dtype=torch.long,
            )

            logits, states_out = step_fn(
                next_token.squeeze(1),
                states,
                pos_tensor,
            )

            # Preserve the tested ESA-Lightning compiled-state boundary.
            states = states_out.clone()

            next_token = sample_next_token(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            generated.append(
                next_token
            )

            if (
                eos_token_id is not None
                and bool(
                    (
                        next_token
                        == int(eos_token_id)
                    ).all()
                )
            ):
                break

            if (
                progress_interval
                and (step + 1)
                % int(progress_interval)
                == 0
            ):
                sync()

                elapsed = (
                    time.perf_counter()
                    - decode_start
                )

                done = step + 1

                print(
                    f"ESA-Lightning "
                    f"{done:,}/{decode_target:,} | "
                    f"{done/max(elapsed, 1e-9):,.2f} tok/s"
                )

        sync()

        decode_seconds = (
            time.perf_counter()
            - decode_start
        )

        total_seconds = (
            time.perf_counter()
            - total_start
        )

        generated_ids = torch.cat(
            generated,
            dim=1,
        )

        sequences = torch.cat(
            [
                input_ids,
                generated_ids,
            ],
            dim=1,
        )

        state_bytes = int(
            states.numel()
            * states.element_size()
        )

        generation_stats = GenerationStats(
            prompt_tokens=prompt_tokens,
            prefill_tokens=int(
                prefill_len
            ),
            generated_tokens=int(
                generated_ids.size(1)
            ),
            decode_steps=max(
                0,
                int(
                    generated_ids.size(1)
                ) - 1,
            ),
            prefill_seconds=prefill_seconds,
            decode_seconds=decode_seconds,
            decode_tok_s=max(
                0,
                int(
                    generated_ids.size(1)
                ) - 1,
            )
            / max(
                decode_seconds,
                1e-9,
            ),
            total_seconds=total_seconds,
            state_bytes=state_bytes,
            state_mb=(
                state_bytes
                / 1024**2
            ),
        )

        result = GenerationResult(
            sequences=sequences,
            generated_ids=generated_ids,
            stats=generation_stats,
        )

        if tokenizer is not None:
            if sequences.size(0) == 1:
                result.text = tokenizer.decode(
                    sequences[0]
                    .detach()
                    .cpu()
                    .tolist()
                )
            else:
                result.text = [
                    tokenizer.decode(
                        row
                        .detach()
                        .cpu()
                        .tolist()
                    )
                    for row in sequences
                ]

        self.train(
            was_training
        )

        if stats:
            return result

        if result.text is not None:
            return result.text

        return sequences

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
