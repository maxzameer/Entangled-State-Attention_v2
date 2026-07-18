# Copyright 2026 Zameer Hussain and Akhtar Hussain
# Licensed under the Apache License, Version 2.0.

import inspect

import pytest
import torch

from esa import ESA


def test_default_backend_is_thunder():
    signature = inspect.signature(
        ESA
    )

    assert (
        signature.parameters[
            "backend"
        ].default
        == "thunder"
    )


def test_thunder_forward_cpu():
    layer = ESA(
        embd=32,
        head=4,
        backend="thunder",
        precision="fp32",
        device=None,
    )

    x = torch.randn(
        2,
        16,
        32,
    )

    y = layer(x)

    assert y.shape == x.shape
    assert layer.compass == 16


def test_pulse_forward_cpu():
    layer = ESA(
        embd=32,
        head=4,
        backend="pulse",
        precision="fp32",
        device=None,
    )

    x = torch.randn(
        2,
        16,
        32,
    )

    y = layer(x)

    assert y.shape == x.shape
    assert layer.compass is None


def test_compass_rejected_for_pulse_and_flare():
    with pytest.raises(
        ValueError,
        match="compass is only supported",
    ):
        ESA(
            embd=32,
            head=4,
            backend="pulse",
            compass=16,
            device=None,
        )

    with pytest.raises(
        ValueError,
        match="compass is only supported",
    ):
        ESA(
            embd=32,
            head=4,
            backend="flare",
            compass=16,
            device=None,
        )


def test_old_dimension_names_are_not_public_api():
    with pytest.raises(
        TypeError,
    ):
        ESA(
            n_embd=32,
            n_head=4,
        )


def test_lightning_is_not_public_backend():
    with pytest.raises(
        ValueError,
        match="Supported backends",
    ):
        ESA(
            embd=32,
            head=4,
            backend="lightning",
            device=None,
        )
