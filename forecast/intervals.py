from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def split_conformal_abs_quantile(residuals: Iterable[float], coverage: float, fallback: float) -> float:
    values = np.asarray(list(residuals), dtype=float)
    values = np.abs(values[np.isfinite(values)])
    if len(values) == 0:
        return float(fallback)

    coverage = float(np.clip(coverage, 0.0, 1.0))
    q_level = min(math.ceil((len(values) + 1) * coverage) / len(values), 1.0)
    try:
        return float(np.quantile(values, q_level, method="higher"))
    except TypeError:
        return float(np.quantile(values, q_level, interpolation="higher"))
