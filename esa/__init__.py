# Copyright 2026 Zameer Hussain and Akhtar Hussain
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

from .config import ESAConfig
from .layer import ESA
from .compass import compass, CompassResult
from .backends import ThunderESA, FlareESA, PulseESA
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

__version__ = "2.0.0"

__all__ = [
    "ESA",
    "ESAConfig",
    "compass",
    "CompassResult",
    "ThunderESA",
    "FlareESA",
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