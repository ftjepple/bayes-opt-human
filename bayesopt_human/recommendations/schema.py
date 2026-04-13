"""Recommendation dataclass."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Recommendation:
    """A ranked recommendation for the next evaluation point."""

    candidate: np.ndarray  # Raw-space coordinates
    candidate_normalized: np.ndarray
    source: str  # Strategy name
    rationale: str  # Plain-language explanation
    confidence: str  # "high", "medium", "low"
    priority: int  # 1 = top recommendation
