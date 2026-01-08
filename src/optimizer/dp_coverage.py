"""Dynamic programming optimizer for full video coverage."""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from ..graph.boundary_graph import BoundaryGraph
from ..graph.candidate_edges import CandidateClip, generate_candidate_edges
from ..graph.scorer import ClipScorer


@dataclass
class CoverageConfig:
    """Configuration for coverage optimization."""
    min_duration: float = 45.0
    max_duration: float = 60.0
    
    # Gap handling
    gap_penalty_per_second: float = 0.5
    overlap_penalty_per_second: float = 1.0
    max_gap_allowed: float = 5.0
    
    # Strict coverage mode
    strict_coverage: bool = False
    strict_max_gap: float = 1.0
    
    # Tail coverage enforcement
    enforce_tail_coverage: bool = True
    tail_penalty_per_second: float = 1.0
    max_tail_gap: float = 10.0


@dataclass 
class OptimizationResult:
    """Result of clip selection optimization."""
    clips: List[CandidateClip]
    total_score: float
    coverage_ratio: float
    total_gap_seconds: float
    tail_gap: float


def optimize_coverage(
    graph: BoundaryGraph,
    scorer: ClipScorer,
    config: CoverageConfig,
    speech_segments: Optional[List[Tuple[float, float]]] = None,
) -> OptimizationResult:
    """
    DP optimization for full video coverage.
    
    Uses backpointers instead of storing full clip lists (O(n) memory).
    Enforces tail coverage (solution must reach near video end).
    """
    if not graph._finalized:
        raise RuntimeError("Call graph.finalize() before optimization")
    
    boundaries = graph.boundaries
    n = len(boundaries)
    video_end = graph.video_duration
    
    if n < 2:
        return OptimizationResult([], 0.0, 0.0, 0.0, video_end)
    
    # === Build edge index: end_idx -> [(clip, score), ...] ===
    edges: Dict[int, List[Tuple[CandidateClip, float]]] = {i: [] for i in range(n)}
    
    for clip in generate_candidate_edges(
        graph,
        config.min_duration,
        config.max_duration,
        speech_segments,
    ):
        score = scorer.score(clip)
        edges[clip.end_idx].append((clip, score))
    
    # === DP with backpointers ===
    NEG_INF = float('-inf')
    
    best_score: List[float] = [NEG_INF] * n
    last_end_time: List[float] = [0.0] * n
    parent: List[Optional[Tuple[int, CandidateClip]]] = [None] * n
    
    # Base case: boundary 0 with no clips, last_end = 0.0
    best_score[0] = 0.0
    last_end_time[0] = 0.0
    parent[0] = None
    
    for end_idx in range(1, n):
        # Option 1: Skip this boundary (carry forward from previous)
        if best_score[end_idx - 1] > best_score[end_idx]:
            best_score[end_idx] = best_score[end_idx - 1]
            last_end_time[end_idx] = last_end_time[end_idx - 1]
            parent[end_idx] = parent[end_idx - 1]
        
        # Option 2: End a clip at this boundary
        for clip, clip_score in edges[end_idx]:
            start_idx = clip.start_idx
            
            if best_score[start_idx] == NEG_INF:
                continue
            
            # Gap from previous clip end to this clip start
            prev_end = last_end_time[start_idx]
            gap = clip.start - prev_end
            
            # Strict coverage check
            if config.strict_coverage and gap > config.strict_max_gap:
                continue
            
            # Max gap check
            if gap > config.max_gap_allowed:
                continue
            
            # Gap/overlap penalty
            if gap > 0:
                transition_penalty = gap * config.gap_penalty_per_second
            else:
                transition_penalty = abs(gap) * config.overlap_penalty_per_second
            
            total = best_score[start_idx] + clip_score - transition_penalty
            
            if total > best_score[end_idx]:
                best_score[end_idx] = total
                last_end_time[end_idx] = clip.end
                parent[end_idx] = (start_idx, clip)
    
    # === Find best terminal state ===
    best_terminal_idx = -1
    best_terminal_score = NEG_INF
    
    for i in range(n):
        if best_score[i] == NEG_INF:
            continue
        
        tail_gap = video_end - last_end_time[i]
        
        # Reject if tail gap too large
        if config.enforce_tail_coverage and tail_gap > config.max_tail_gap:
            continue
        
        # Apply tail penalty
        adjusted_score = best_score[i]
        if config.enforce_tail_coverage and tail_gap > 0:
            adjusted_score -= tail_gap * config.tail_penalty_per_second
        
        if adjusted_score > best_terminal_score:
            best_terminal_score = adjusted_score
            best_terminal_idx = i
    
    # === Reconstruct path via backpointers ===
    if best_terminal_idx < 0:
        return OptimizationResult([], 0.0, 0.0, 0.0, video_end)
    
    clips = _reconstruct_path(parent, best_terminal_idx)
    
    # === Compute stats ===
    if clips:
        total_clip_time = sum(c.duration for c in clips)
        coverage_ratio = total_clip_time / video_end if video_end > 0 else 0.0
        
        total_gap = 0.0
        prev_end = 0.0
        for clip in clips:
            gap = clip.start - prev_end
            if gap > 0:
                total_gap += gap
            prev_end = clip.end
        
        tail_gap = video_end - clips[-1].end
    else:
        coverage_ratio = 0.0
        total_gap = 0.0
        tail_gap = video_end
    
    return OptimizationResult(
        clips=clips,
        total_score=best_terminal_score if best_terminal_score > NEG_INF else 0.0,
        coverage_ratio=coverage_ratio,
        total_gap_seconds=total_gap,
        tail_gap=tail_gap,
    )


def _reconstruct_path(
    parent: List[Optional[Tuple[int, CandidateClip]]],
    terminal_idx: int,
) -> List[CandidateClip]:
    """
    Walk backpointers to reconstruct clip sequence.
    
    O(k) where k = number of clips in solution.
    """
    clips = []
    idx = terminal_idx
    
    while idx is not None and parent[idx] is not None:
        prev_idx, clip = parent[idx]
        clips.append(clip)
        idx = prev_idx
    
    clips.reverse()
    return clips
