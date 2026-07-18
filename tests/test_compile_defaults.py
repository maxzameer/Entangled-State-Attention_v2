from __future__ import annotations

import inspect

from esa.model import ESAModel, ESAModelConfig


def test_training_compile_mode_default() -> None:
    config = ESAModelConfig(vocab_size=128)
    assert config.training_compile is True
    assert config.training_compile_mode == "default"


def test_prefill_compile_mode_default() -> None:
    signature = inspect.signature(ESAModel.prefill)
    assert signature.parameters["compile_mode"].default == "default"


def test_compile_generation_mode_default() -> None:
    signature = inspect.signature(ESAModel.compile_generation)
    assert signature.parameters["mode"].default == "default"


def test_generate_compile_mode_default() -> None:
    signature = inspect.signature(ESAModel.generate)
    assert signature.parameters["compile"].default is True
    assert signature.parameters["compile_mode"].default == "default"
