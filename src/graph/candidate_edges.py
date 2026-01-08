"""Candidate clip edge generation."""

from dataclasses import dataclass
from typing import List, Iterator, Optional, Tuple

from .boundary_graph import BoundaryGraph, Boundary, BoundaryType, StitchBehavior


@dataclass
class CandidateClip:
    """An edge in the DAG: a potential clip from start to end."""
    start_idx: int
    end_idx: int
    start_boundary: Boundary
    end_boundary: Boundary
    
    # Precomputed features
    duration: float = 0.0
    has_speech_in_first_500ms: bool = True
    silence_ratio: float = 0.0
    contains_forbidden_stitch: bool = False
    max_speech_density_at_boundaries: float = 0.0
    
    # Stitch alignment
    starts_at_stitch: bool = False
    ends_at_stitch: bool = False
    
    @property
    def start(self) -> float:
        return self.start_boundary.timestamp
    
    @property
    def end(self) -> float:
        return self.end_boundary.timestamp


class ClipFeatureComputer:
    """
    Computes features for candidate clips.
    
    Uses prefix sum for O(1) speech queries.
    """
    
    def __init__(
        self,
        graph: BoundaryGraph,
        speech_segments: Optional[List[Tuple[float, float]]] = None,
    ):
        self.graph = graph
        self.speech_segments = sorted(speech_segments or [], key=lambda x: x[0])
        
        # Build prefix sum to video_duration
        self._grid_size = 0.05  # 50ms buckets
        self._speech_prefix = self._build_speech_prefix()
    
    def _build_speech_prefix(self) -> List[float]:
        """
        Build prefix sum of speech time at 50ms granularity.
        
        Extends to video_duration, not just max(seg_end).
        """
        if not self.speech_segments and self.graph.video_duration <= 0:
            return []
        
        # Use video duration as max time
        max_time = self.graph.video_duration
        if self.speech_segments:
            max_time = max(max_time, max(seg[1] for seg in self.speech_segments))
        
        n_buckets = int(max_time / self._grid_size) + 2
        
        bucket_speech = [0.0] * n_buckets
        
        for seg_start, seg_end in self.speech_segments:
            start_bucket = int(seg_start / self._grid_size)
            end_bucket = int(seg_end / self._grid_size) + 1
            
            for b in range(start_bucket, min(end_bucket, n_buckets)):
                bucket_start = b * self._grid_size
                bucket_end = (b + 1) * self._grid_size
                
                overlap_start = max(seg_start, bucket_start)
                overlap_end = min(seg_end, bucket_end)
                
                if overlap_end > overlap_start:
                    bucket_speech[b] += overlap_end - overlap_start
        
        prefix = [0.0] * (n_buckets + 1)
        for i in range(n_buckets):
            prefix[i + 1] = prefix[i] + bucket_speech[i]
        
        return prefix
    
    def compute(self, start_idx: int, end_idx: int) -> CandidateClip:
        """Compute all features for a clip."""
        start_boundary = self.graph.boundaries[start_idx]
        end_boundary = self.graph.boundaries[end_idx]
        start_time = start_boundary.timestamp
        end_time = end_boundary.timestamp
        duration = end_time - start_time
        
        clip = CandidateClip(
            start_idx=start_idx,
            end_idx=end_idx,
            start_boundary=start_boundary,
            end_boundary=end_boundary,
            duration=duration,
        )
        
        # Stitch features
        clip.contains_forbidden_stitch = self.graph.contains_forbidden_stitch(
            start_time, end_time
        )
        clip.starts_at_stitch = self.graph.is_stitch_boundary(start_time)
        clip.ends_at_stitch = self.graph.is_stitch_boundary(end_time)
        
        # Speech features
        if self._speech_prefix:
            clip.has_speech_in_first_500ms = self._has_speech_in_range_fast(
                start_time, start_time + 0.5
            )
            clip.silence_ratio = self._compute_silence_ratio_fast(start_time, end_time)
        
        clip.max_speech_density_at_boundaries = max(
            start_boundary.speech_density or 0.0,
            end_boundary.speech_density or 0.0,
        )
        
        return clip
    
    def recompute(self, clip: CandidateClip) -> CandidateClip:
        """Recompute features for an existing clip."""
        return self.compute(clip.start_idx, clip.end_idx)
    
    def _has_speech_in_range_fast(self, start: float, end: float) -> bool:
        """O(1) check using prefix sum."""
        if not self._speech_prefix:
            return True
        
        start_bucket = int(start / self._grid_size)
        end_bucket = int(end / self._grid_size) + 1
        
        # Clamp to prefix bounds
        start_bucket = max(0, min(start_bucket, len(self._speech_prefix) - 1))
        end_bucket = max(0, min(end_bucket, len(self._speech_prefix) - 1))
        
        speech_time = self._speech_prefix[end_bucket] - self._speech_prefix[start_bucket]
        return speech_time > 0.01
    
    def _compute_silence_ratio_fast(self, start: float, end: float) -> float:
        """O(1) silence ratio using prefix sum."""
        duration = end - start
        if duration <= 0:
            return 0.0
        
        if not self._speech_prefix:
            return 0.0
        
        start_bucket = int(start / self._grid_size)
        end_bucket = int(end / self._grid_size) + 1
        
        # Clamp to prefix bounds
        start_bucket = max(0, min(start_bucket, len(self._speech_prefix) - 1))
        end_bucket = max(0, min(end_bucket, len(self._speech_prefix) - 1))
        
        speech_time = self._speech_prefix[end_bucket] - self._speech_prefix[start_bucket]
        return max(0.0, min(1.0, 1.0 - (speech_time / duration)))


def generate_candidate_edges(
    graph: BoundaryGraph,
    min_duration: float = 45.0,
    max_duration: float = 60.0,
    speech_segments: Optional[List[Tuple[float, float]]] = None,
) -> Iterator[CandidateClip]:
    """
    Generate all valid clip candidates.
    
    Uses O(log m) has_stitch_interior() for MUST_CUT.
    Windowed iteration: O(n * k) where k = avg boundaries in duration window.
    """
    if not graph._finalized:
        raise RuntimeError("Call graph.finalize() before generating edges")
    
    boundaries = graph.boundaries
    n = len(boundaries)
    
    if n < 2:
        return
    
    feature_computer = ClipFeatureComputer(graph, speech_segments)
    must_cut = graph.stitch_behavior == StitchBehavior.MUST_CUT
    
    window_start = 0
    
    for end_idx in range(1, n):
        end_boundary = boundaries[end_idx]
        end_time = end_boundary.timestamp
        
        # MUST_CUT: end must be at stitch or video end
        if must_cut:
            is_valid_end = (
                end_boundary.has_type(BoundaryType.STITCH_MARK) or
                end_boundary.has_type(BoundaryType.VIDEO_END)
            )
            if not is_valid_end:
                continue
        
        min_start_time = end_time - max_duration
        max_start_time = end_time - min_duration
        
        while (window_start < end_idx and 
               boundaries[window_start].timestamp < min_start_time):
            window_start += 1
        
        for start_idx in range(window_start, end_idx):
            start_boundary = boundaries[start_idx]
            start_time = start_boundary.timestamp
            
            if start_time > max_start_time:
                break
            
            duration = end_time - start_time
            if not (min_duration <= duration <= max_duration):
                continue
            
            # MUST_CUT: start must be at stitch or video start
            if must_cut:
                is_valid_start = (
                    start_boundary.has_type(BoundaryType.STITCH_MARK) or
                    start_boundary.has_type(BoundaryType.VIDEO_START)
                )
                if not is_valid_start:
                    continue
                
                # No interior stitch marks — O(log m)
                if graph.has_stitch_interior(start_time, end_time):
                    continue
            
            yield feature_computer.compute(start_idx, end_idx)
