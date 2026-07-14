# Copyright 2026 Zameer Hussain and Akhtar Hussain
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

from .config import ESAConfig
from .layer import ESA
from .model import ESAModel, ESAModelConfig
from .trainer import Trainer, TrainerState
from .generation import GenerationResult, GenerationStats
from .compass import compass, CompassResult
from .backends import (
    FlareESA,
    ThunderESA,
    PulseESA,
)
from .benchmark import (
    ESABenchmarkConfig,
    DEFAULT_BENCHMARK_CONFIG,
    FAST_BENCHMARK_CONFIG,
    PAPER_BENCHMARK_CONFIG,
    BENCHMARK_DEFAULTS,
    FAST_BENCHMARK_DEFAULTS,
    PAPER_BENCHMARK_DEFAULTS,
)
from .boost import thunderBoost

__version__ = "2.1.1"

__all__ = [
    "ESA",
    "ESAConfig",
    "ESAModel",
    "ESAModelConfig",
    "Trainer",
    "TrainerState",
    "GenerationResult",
    "GenerationStats",
    "compass",
    "CompassResult",
    "FlareESA",
    "ThunderESA",
    "PulseESA",
    "thunderBoost",
    "ESABenchmarkConfig",
    "DEFAULT_BENCHMARK_CONFIG",
    "FAST_BENCHMARK_CONFIG",
    "PAPER_BENCHMARK_CONFIG",
    "BENCHMARK_DEFAULTS",
    "FAST_BENCHMARK_DEFAULTS",
    "PAPER_BENCHMARK_DEFAULTS",
]
