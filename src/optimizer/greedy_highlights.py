"""Greedy highlights selection for top-k non-overlapping clips."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..graph.boundary_graph import BoundaryGraph
from ..graph.candidate_edges import CandidateClip, generate_candidate_edges
from ..graph.scorer import ClipScorer


@dataclass
class HighlightsConfig:
    """Configuration for highlights selection."""
    min_duration: float = 45.0
    max_duration: float = 60.0
    max_clips: int = 10
    min_gap_between_clips: float = 0.0


def select_highlights(
    graph: BoundaryGraph,
    scorer: ClipScorer,
    config: HighlightsConfig,
    speech_segments: Optional[List[Tuple[float, float]]] = None,
) -> List[CandidateClip]:
    """
    Select top-k non-overlapping clips by score.
    
    Algorithm:
    1. Generate all candidates, score them
    2. Sort by score descending
    3. Greedily pick highest-scoring non-overlapping clips
    """
    if not graph._finalized:
        raise RuntimeError("Call graph.finalize() before selection")
    
    # Generate and score all candidates
    scored_clips: List[Tuple[float, CandidateClip]] = []
    
    for clip in generate_candidate_edges(
        graph,
        config.min_duration,
        config.max_duration,
        speech_segments,
    ):
        score = scorer.score(clip)
        scored_clips.append((score, clip))
    
    # Sort by score descending
    scored_clips.sort(key=lambda x: x[0], reverse=True)
    
    # Greedy non-overlapping selection
    selected: List[CandidateClip] = []
    
    for score, clip in scored_clips:
        if len(selected) >= config.max_clips:
            break
        
        # Check for overlap with already selected
        overlaps = False
        for existing in selected:
            if _clips_overlap(clip, existing, config.min_gap_between_clips):
                overlaps = True
                break
        
        if not overlaps:
            selected.append(clip)
    
    # Sort by start time for output
    selected.sort(key=lambda c: c.start)
    
    return selected


def _clips_overlap(
    a: CandidateClip, 
    b: CandidateClip, 
    min_gap: float
) -> bool:
    """Check if two clips overlap (with min_gap buffer)."""
    return not (a.end + min_gap <= b.start or b.end + min_gap <= a.start)
