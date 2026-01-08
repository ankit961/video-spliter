"""Boundary graph data structures."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set, Dict
import bisect


class BoundaryType(Enum):
    """Types of boundaries that can be detected."""
    SHOT = "shot"
    VAD_PAUSE = "vad_pause"
    SENTENCE_END = "sentence_end"
    STITCH_MARK = "stitch_mark"
    ENERGY_DIP = "energy_dip"
    SPEAKER_TURN = "speaker_turn"
    VIDEO_START = "video_start"
    VIDEO_END = "video_end"


class StitchBehavior(Enum):
    """How to treat stitch marks in optimization."""
    FORBIDDEN_INTERIOR = "forbidden_interior"  # Cannot appear mid-clip
    PREFERRED_CUT = "preferred_cut"            # Bonus when used as start/end
    MUST_CUT = "must_cut"                      # Clips must align to these


# Anchor types that must preserve their exact timestamp during merge
ANCHOR_PRIORITY: Dict[BoundaryType, int] = {
    BoundaryType.VIDEO_START: 3,
    BoundaryType.VIDEO_END: 3,
    BoundaryType.STITCH_MARK: 2,
}


def _anchor_priority(b: "Boundary") -> int:
    """Get highest anchor priority from boundary types."""
    return max((ANCHOR_PRIORITY.get(t, 0) for t in b.types), default=0)


@dataclass
class Boundary:
    """
    A single point where we CAN cut.
    
    Supports multiple types at the same timestamp (e.g., pause + sentence_end).
    """
    timestamp: float
    types: Set[BoundaryType] = field(default_factory=set)
    confidences: Dict[BoundaryType, float] = field(default_factory=dict)
    
    # Optional metadata
    pause_duration: Optional[float] = None
    energy_slope: Optional[float] = None
    speech_density: Optional[float] = None
    
    @property
    def confidence(self) -> float:
        """Get highest confidence across all types."""
        if not self.confidences:
            return 1.0
        return max(self.confidences.values())
    
    def __lt__(self, other):
        return self.timestamp < other.timestamp
    
    def merge_from(self, other: "Boundary"):
        """
        Merge another boundary's data into this one.
        
        Preserves anchor timestamps (VIDEO_START/END, STITCH_MARK)
        to prevent drift during nearby-merge.
        """
        # === Anchor timestamp preservation ===
        self_pri = _anchor_priority(self)
        other_pri = _anchor_priority(other)
        
        if other_pri > self_pri:
            # Other has higher priority anchor — use its timestamp
            self.timestamp = other.timestamp
        elif other_pri == self_pri and other_pri > 0:
            # Same priority anchors — prefer higher confidence
            if BoundaryType.STITCH_MARK in other.types:
                self_conf = self.confidences.get(BoundaryType.STITCH_MARK, 0.0)
                other_conf = other.confidences.get(BoundaryType.STITCH_MARK, 0.0)
                if other_conf > self_conf:
                    self.timestamp = other.timestamp
        
        # === Merge types and confidences ===
        self.types.update(other.types)
        
        for btype, conf in other.confidences.items():
            if btype not in self.confidences or conf > self.confidences[btype]:
                self.confidences[btype] = conf
        
        # === Merge metadata (keep best) ===
        if other.pause_duration is not None:
            if self.pause_duration is None:
                self.pause_duration = other.pause_duration
            else:
                self.pause_duration = max(self.pause_duration, other.pause_duration)
        
        if other.energy_slope is not None:
            self.energy_slope = other.energy_slope
        
        if other.speech_density is not None:
            self.speech_density = other.speech_density
    
    def has_type(self, btype: BoundaryType) -> bool:
        """Check if boundary has a specific type."""
        return btype in self.types
    
    @property
    def is_multi_signal(self) -> bool:
        """True if multiple real detection methods agree."""
        real_signals = self.types - {BoundaryType.VIDEO_START, BoundaryType.VIDEO_END}
        return len(real_signals) > 1
    
    @property
    def is_anchor(self) -> bool:
        """True if this boundary has an anchor type."""
        return _anchor_priority(self) > 0


@dataclass
class BoundaryGraph:
    """
    All valid cut points, sorted by time.
    
    Nodes = boundaries, Edges = valid clips between them.
    """
    boundaries: List[Boundary] = field(default_factory=list)
    video_duration: float = 0.0
    
    stitch_marks: List[float] = field(default_factory=list)
    stitch_behavior: StitchBehavior = StitchBehavior.FORBIDDEN_INTERIOR
    
    _sorted_stitch_marks: List[float] = field(default_factory=list)
    _finalized: bool = False
    
    def add(self, boundary: Boundary):
        """Add a boundary (will be sorted on finalize)."""
        if self._finalized:
            raise RuntimeError("Cannot add boundaries after finalize()")
        self.boundaries.append(boundary)
    
    def add_stitch_mark(self, timestamp: float):
        """Add a stitch mark."""
        if self._finalized:
            raise RuntimeError("Cannot add stitch marks after finalize()")
        self.stitch_marks.append(timestamp)
    
    def finalize(
        self,
        merge_threshold: float = 0.3,
        downsample_sentence_gap: float = 0.5,
    ):
        """
        Finalize the graph for use.
        
        Canonical order:
        1. Sort
        2. Inject VIDEO_START/END
        3. Inject stitch marks as boundaries
        4. Sort
        5. Merge nearby (with anchor preservation)
        6. Downsample dense sentence boundaries
        7. Final sort
        8. Build stitch index
        9. Lock
        """
        if self._finalized:
            return
        
        # 1. Initial sort
        self.boundaries.sort(key=lambda b: b.timestamp)
        
        # 2. Inject VIDEO_START/END
        self._ensure_endpoint(0.0, BoundaryType.VIDEO_START)
        self._ensure_endpoint(self.video_duration, BoundaryType.VIDEO_END)
        
        # 3. Inject stitch marks
        for t in self.stitch_marks:
            self._ensure_stitch_boundary(t)
        
        # 4. Sort after injections
        self.boundaries.sort(key=lambda b: b.timestamp)
        
        # 5. Merge nearby (anchor-aware)
        self._merge_nearby(merge_threshold)
        
        # 6. Downsample sentence boundaries
        if downsample_sentence_gap > 0:
            self._downsample_by_type(BoundaryType.SENTENCE_END, downsample_sentence_gap)
        
        # 7. Final sort (merge may have reordered due to anchor snapping)
        self.boundaries.sort(key=lambda b: b.timestamp)
        
        # 8. Build sorted stitch index
        self._sorted_stitch_marks = sorted(set(self.stitch_marks))
        
        # 9. Lock
        self._finalized = True
    
    def _ensure_endpoint(self, timestamp: float, btype: BoundaryType):
        """Ensure a boundary exists at exact timestamp."""
        for b in self.boundaries:
            if abs(b.timestamp - timestamp) < 0.05:
                b.types.add(btype)
                b.confidences[btype] = 1.0
                return
        
        self.boundaries.append(Boundary(
            timestamp=timestamp,
            types={btype},
            confidences={btype: 1.0},
        ))
    
    def _ensure_stitch_boundary(self, timestamp: float):
        """Ensure a stitch mark exists as boundary."""
        for b in self.boundaries:
            if abs(b.timestamp - timestamp) < 0.1:
                b.types.add(BoundaryType.STITCH_MARK)
                b.confidences[BoundaryType.STITCH_MARK] = 1.0
                return
        
        self.boundaries.append(Boundary(
            timestamp=timestamp,
            types={BoundaryType.STITCH_MARK},
            confidences={BoundaryType.STITCH_MARK: 1.0},
        ))
    
    def _merge_nearby(self, threshold: float):
        """
        Merge boundaries within threshold seconds.
        
        Anchor timestamps are preserved during merge.
        """
        if not self.boundaries:
            return
        
        merged = [self.boundaries[0]]
        
        for b in self.boundaries[1:]:
            if b.timestamp - merged[-1].timestamp < threshold:
                merged[-1].merge_from(b)
            else:
                merged.append(b)
        
        self.boundaries = merged
    
    def _downsample_by_type(self, btype: BoundaryType, min_gap: float):
        """Reduce density of a specific boundary type."""
        buckets: Dict[int, List[Boundary]] = {}
        
        for b in self.boundaries:
            if btype in b.types:
                bucket_idx = int(b.timestamp / min_gap)
                if bucket_idx not in buckets:
                    buckets[bucket_idx] = []
                buckets[bucket_idx].append(b)
        
        for bucket_boundaries in buckets.values():
            if len(bucket_boundaries) <= 1:
                continue
            
            bucket_boundaries.sort(
                key=lambda b: b.confidences.get(btype, 0),
                reverse=True
            )
            
            for b in bucket_boundaries[1:]:
                b.types.discard(btype)
                b.confidences.pop(btype, None)
        
        self.boundaries = [b for b in self.boundaries if b.types]
    
    def has_stitch_interior(self, start: float, end: float) -> bool:
        """
        O(log m) check for stitch mark strictly inside (start, end).
        
        Used for MUST_CUT validation.
        """
        if not self._sorted_stitch_marks:
            return False
        
        i = bisect.bisect_right(self._sorted_stitch_marks, start)
        return i < len(self._sorted_stitch_marks) and self._sorted_stitch_marks[i] < end
    
    def contains_forbidden_stitch(self, start: float, end: float) -> bool:
        """O(log m) check for forbidden interior stitch."""
        if self.stitch_behavior != StitchBehavior.FORBIDDEN_INTERIOR:
            return False
        return self.has_stitch_interior(start, end)
    
    def is_stitch_boundary(self, timestamp: float, epsilon: float = 0.1) -> bool:
        """Check if a timestamp is at a stitch mark."""
        if not self._sorted_stitch_marks:
            return False
        
        idx = bisect.bisect_left(self._sorted_stitch_marks, timestamp - epsilon)
        if idx < len(self._sorted_stitch_marks):
            return abs(self._sorted_stitch_marks[idx] - timestamp) < epsilon
        return False
    
    def get_boundaries_in_range(self, start: float, end: float) -> List[Boundary]:
        """Binary search for boundaries in [start, end]."""
        if not self._finalized:
            raise RuntimeError("Call finalize() before querying")
        
        left = bisect.bisect_left(
            self.boundaries,
            Boundary(timestamp=start, types=set())
        )
        right = bisect.bisect_right(
            self.boundaries,
            Boundary(timestamp=end, types=set())
        )
        return self.boundaries[left:right]
    
    def get_index(self, timestamp: float, epsilon: float = 0.05) -> Optional[int]:
        """Get boundary index by timestamp."""
        idx = bisect.bisect_left(
            self.boundaries,
            Boundary(timestamp=timestamp, types=set())
        )
        if idx < len(self.boundaries) and abs(self.boundaries[idx].timestamp - timestamp) < epsilon:
            return idx
        return None
