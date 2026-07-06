# Copyright 2026 Zameer Hussain and Akhtar Hussain
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

ESA_GATE_MIN = 0.80
ESA_GATE_MAX = 0.995
ESA_SCAN_EPS = 1e-6
SUPPORTED_BACKENDS = {"thunder", "flare", "pulse"}
SUPPORTED_PRECISIONS = {"fp8", "fp16", "bf16", "fp32", "fp64"}
