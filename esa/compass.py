# Copyright 2026 Zameer Hussain and Akhtar Hussain
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable


@dataclass
class CompassResult:
    """Result returned by ``compass()``.

    The class is about Thunder's ``c``, so the fields are intentionally short:

    - recommended: best practical c.
    - best_quality: c with best validation loss/PPL.
    - fastest: c with highest tokens/sec.
    """

    recommended: int
    best_quality: int
    fastest: int
    rows: list[dict[str, Any]]
    reference_backend: str | None
    precision: str
    quality_tolerance: float
    recommendation: str

    def summary(self) -> str:
        return self.recommendation

    def to_dataframe(self):
        try:
            import pandas as pd
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("pandas is required for to_dataframe(). Install pandas first.") from exc
        return pd.DataFrame(self.rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommended": self.recommended,
            "best_quality": self.best_quality,
            "fastest": self.fastest,
            "rows": self.rows,
            "reference_backend": self.reference_backend,
            "precision": self.precision,
            "quality_tolerance": self.quality_tolerance,
            "recommendation": self.recommendation,
        }


def _as_float(row: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = row.get(name)
        if value is not None:
            try:
                return float(value)
            except Exception:
                return None
    return None


def _normalise_row(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    if "ppl" not in row and "val_loss" in row:
        try:
            row["ppl"] = float(math.exp(float(row["val_loss"])))
        except Exception:
            pass
    if "tok_per_sec" not in row:
        for alias in ("tokens_per_second", "tok_s", "throughput", "tokens_sec"):
            if alias in row:
                row["tok_per_sec"] = row[alias]
                break
    return row


def compass(
    *,
    evaluate_fn: Callable[..., dict[str, Any]],
    c_candidates: Iterable[int] = (8, 16, 32),
    reference_backend: str | None = None,
    precision: str = "fp16",
    quality_tolerance: float = 0.02,
    metric: str = "ppl",
    speed_metric: str = "tok_per_sec",
    **evaluate_kwargs: Any,
) -> CompassResult:
    """Select a practical Thunder ``c`` for a workload.

    ``compass()`` evaluates Thunder ESA across candidate ``c`` values and
    recommends the fastest c whose quality remains close to the best observed
    result.

    The function is intentionally framework-light. The caller supplies
    ``evaluate_fn`` so Compass can work with any training loop.

    ``evaluate_fn`` is called as::

        evaluate_fn(backend="thunder", c=<candidate>, precision=precision, **kwargs)

    It should return a dict containing at least:

        - val_loss or ppl
        - tok_per_sec, tokens_per_second, tok_s, throughput, or tokens_sec

    If ``reference_backend`` is "pulse" or "flare", Compass also calls::

        evaluate_fn(backend=reference_backend, c=None, precision=precision, **kwargs)

    and stores that reference row for comparison. The recommendation is still
    always a Thunder c, because only Thunder exposes c.
    """
    c_values = tuple(int(c) for c in c_candidates)
    if not c_values:
        raise ValueError("c_candidates must contain at least one value.")
    if any(c <= 0 for c in c_values):
        raise ValueError(f"all c values must be positive integers, got {c_values}")
    if reference_backend not in {None, "pulse", "flare"}:
        raise ValueError('reference_backend must be None, "pulse", or "flare".')
    if quality_tolerance < 0:
        raise ValueError("quality_tolerance must be >= 0.")

    rows: list[dict[str, Any]] = []

    if reference_backend is not None:
        ref_row = evaluate_fn(
            backend=reference_backend,
            c=None,
            precision=precision,
            **evaluate_kwargs,
        )
        ref_row = _normalise_row(ref_row)
        ref_row.setdefault("backend", reference_backend)
        ref_row.setdefault("c", None)
        ref_row["is_reference"] = True
        rows.append(ref_row)

    thunder_rows: list[dict[str, Any]] = []
    for c in c_values:
        row = evaluate_fn(
            backend="thunder",
            c=c,
            precision=precision,
            **evaluate_kwargs,
        )
        row = _normalise_row(row)
        row.setdefault("backend", "thunder")
        row["c"] = c
        row["is_reference"] = False
        thunder_rows.append(row)
        rows.append(row)

    def quality_value(row: dict[str, Any]) -> float:
        value = _as_float(row, metric, "ppl", "val_loss")
        if value is None:
            raise ValueError(
                f"Each evaluation row must contain {metric!r}, 'ppl', or 'val_loss'. Bad row: {row}"
            )
        return value

    def speed_value(row: dict[str, Any]) -> float:
        value = _as_float(row, speed_metric, "tok_per_sec", "tokens_per_second", "tok_s", "throughput", "tokens_sec")
        if value is None:
            raise ValueError(
                f"Each Thunder row must contain {speed_metric!r} or a supported speed alias. Bad row: {row}"
            )
        return value

    best_quality_row = min(thunder_rows, key=quality_value)
    fastest_row = max(thunder_rows, key=speed_value)

    best_quality_val = quality_value(best_quality_row)
    threshold = best_quality_val * (1.0 + quality_tolerance)
    acceptable = [row for row in thunder_rows if quality_value(row) <= threshold]
    recommended_row = max(acceptable, key=speed_value)

    for row in thunder_rows:
        row["acceptable"] = row in acceptable

    recommended = int(recommended_row["c"])
    best_quality = int(best_quality_row["c"])
    fastest = int(fastest_row["c"])

    recommendation = (
        f"Use c={recommended}. It is the fastest Thunder c within "
        f"{quality_tolerance * 100:.2f}% of the best observed {metric}. "
        f"Best quality c={best_quality}; fastest c={fastest}."
    )

    return CompassResult(
        recommended=recommended,
        best_quality=best_quality,
        fastest=fastest,
        rows=rows,
        reference_backend=reference_backend,
        precision=precision,
        quality_tolerance=quality_tolerance,
        recommendation=recommendation,
    )
