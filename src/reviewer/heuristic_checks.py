"""Heuristic checks for clip quality assessment."""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

from ..graph.candidate_edges import CandidateClip


class HeuristicResult(Enum):
    """Result of heuristic checks."""
    PASS = "pass"
    FAIL = "fail"
    NEEDS_LLM = "needs_llm"


@dataclass
class HeuristicConfig:
    """Configuration for heuristic checks."""
    max_duration: float = 60.0
    min_duration: float = 45.0
    max_silence_ratio: float = 0.20
    require_speech_in_first_ms: float = 500.0
    max_speech_density_at_cut: float = 4.0  # words/sec
    min_boundary_confidence: float = 0.5


@dataclass
class HeuristicCheckResult:
    """Result of running heuristic checks."""
    result: HeuristicResult
    failed_checks: List[str]
    passed_checks: List[str]


def run_heuristic_checks(
    clip: CandidateClip,
    config: Optional[HeuristicConfig] = None,
) -> HeuristicCheckResult:
    """
    Fast heuristic checks before LLM review.
    
    Returns PASS if all checks pass.
    Returns FAIL if any hard failure.
    Returns NEEDS_LLM if borderline.
    """
    config = config or HeuristicConfig()
    
    failed = []
    passed = []
    borderline = []
    
    # === Hard checks (FAIL if violated) ===
    
    # Duration
    if clip.duration > config.max_duration:
        failed.append(f"duration_too_long: {clip.duration:.1f}s > {config.max_duration}s")
    elif clip.duration < config.min_duration:
        failed.append(f"duration_too_short: {clip.duration:.1f}s < {config.min_duration}s")
    else:
        passed.append("duration_ok")
    
    # Hard stitch inside
    if clip.contains_forbidden_stitch:
        failed.append("contains_hard_stitch")
    else:
        passed.append("no_hard_stitch")
    
    # === Soft checks (borderline if violated) ===
    
    # Hook
    if not clip.has_speech_in_first_500ms:
        borderline.append("no_speech_in_first_500ms")
    else:
        passed.append("has_hook")
    
    # Silence
    if clip.silence_ratio > config.max_silence_ratio:
        borderline.append(f"high_silence_ratio: {clip.silence_ratio:.2f}")
    else:
        passed.append("silence_ok")
    
    # Speech density at boundaries
    if clip.max_speech_density_at_boundaries > config.max_speech_density_at_cut:
        borderline.append(f"high_speech_density_at_cut: {clip.max_speech_density_at_boundaries:.1f}")
    else:
        passed.append("clean_cut_points")
    
    # Boundary confidence
    min_conf = min(
        clip.start_boundary.confidence,
        clip.end_boundary.confidence
    )
    if min_conf < config.min_boundary_confidence:
        borderline.append(f"low_boundary_confidence: {min_conf:.2f}")
    else:
        passed.append("confident_boundaries")
    
    # === Decision ===
    if failed:
        return HeuristicCheckResult(
            result=HeuristicResult.FAIL,
            failed_checks=failed,
            passed_checks=passed,
        )
    elif borderline:
        return HeuristicCheckResult(
            result=HeuristicResult.NEEDS_LLM,
            failed_checks=borderline,
            passed_checks=passed,
        )
    else:
        return HeuristicCheckResult(
            result=HeuristicResult.PASS,
            failed_checks=[],
            passed_checks=passed,
        )
