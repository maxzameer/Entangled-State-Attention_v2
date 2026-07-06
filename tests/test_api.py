from __future__ import annotations

import pytest
import torch

from esa import ESA, compass
from esa.backends import ThunderESA, PulseESA


def test_thunder_forward_cpu():
    layer = ESA(n_embd=32, n_head=4)
    x = torch.randn(2, 16, 32)
    y = layer(x)
    assert y.shape == x.shape
    assert layer.backend == "thunder"
    assert layer.c == 16


def test_pulse_forward_cpu():
    layer = ESA(n_embd=32, n_head=4, backend="pulse")
    x = torch.randn(2, 16, 32)
    y = layer(x)
    assert y.shape == x.shape
    assert layer.c is None


def test_c_rejected_for_pulse_and_flare():
    with pytest.raises(ValueError, match="c is only supported"):
        ESA(n_embd=32, n_head=4, backend="pulse", c=16)


def test_direct_imports():
    assert ThunderESA(n_embd=32, n_head=4, c=16)
    assert PulseESA(n_embd=32, n_head=4)


def test_compass():
    def evaluate_fn(*, backend, c, precision):
        table = {
            8: {"val_loss": 1.91, "tok_per_sec": 10.0},
            16: {"val_loss": 1.88, "tok_per_sec": 11.0},
            32: {"val_loss": 1.895, "tok_per_sec": 12.0},
        }
        if backend == "pulse":
            return {"val_loss": 1.88, "tok_per_sec": 5.0}
        return table[c]

    result = compass(
        evaluate_fn=evaluate_fn,
        c_candidates=(8, 16, 32),
        reference_backend="pulse",
        quality_tolerance=0.02,
    )
    assert result.best_quality == 16
    assert result.fastest == 32
    assert result.recommended == 32
