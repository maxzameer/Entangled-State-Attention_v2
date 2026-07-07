from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ESABenchmarkConfig:
    """
    Benchmark settings for ESA experiments.

    These settings are separate from the normal ESA layer.
    ESA layers do not automatically warm up, compile, train, or benchmark.
    """

    compile_warmup_steps: int = 2
    speed_warmup_steps: int = 2
    speed_bench_steps: int = 10
    compile_mode: str = "reduce-overhead"
    reset_seed_after_compile_warmup: bool = True


DEFAULT_BENCHMARK_CONFIG = ESABenchmarkConfig(
    compile_warmup_steps=2,
    speed_warmup_steps=2,
    speed_bench_steps=10,
    compile_mode="reduce-overhead",
    reset_seed_after_compile_warmup=True,
)


FAST_BENCHMARK_CONFIG = ESABenchmarkConfig(
    compile_warmup_steps=1,
    speed_warmup_steps=1,
    speed_bench_steps=6,
    compile_mode="reduce-overhead",
    reset_seed_after_compile_warmup=True,
)


PAPER_BENCHMARK_CONFIG = ESABenchmarkConfig(
    compile_warmup_steps=3,
    speed_warmup_steps=5,
    speed_bench_steps=30,
    compile_mode="reduce-overhead",
    reset_seed_after_compile_warmup=True,
)


# Dict aliases for users who prefer dictionary-style access.
BENCHMARK_DEFAULTS = asdict(DEFAULT_BENCHMARK_CONFIG)
FAST_BENCHMARK_DEFAULTS = asdict(FAST_BENCHMARK_CONFIG)
PAPER_BENCHMARK_DEFAULTS = asdict(PAPER_BENCHMARK_CONFIG)