"""Optimizer module for clip selection."""

from .dp_coverage import (
    optimize_coverage,
    CoverageConfig,
    OptimizationResult,
)
from .greedy_highlights import (
    select_highlights,
    HighlightsConfig,
)

__all__ = [
    "optimize_coverage",
    "CoverageConfig",
    "OptimizationResult",
    "select_highlights",
    "HighlightsConfig",
]
