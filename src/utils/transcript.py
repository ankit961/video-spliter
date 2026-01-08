"""Transcript data structures with optimized range queries."""

from dataclasses import dataclass, field
from typing import List, Tuple
import bisect


@dataclass
class TranscriptSegment:
    """A segment of transcript with timing."""
    start: float
    end: float
    text: str
    
    def __lt__(self, other):
        if isinstance(other, TranscriptSegment):
            return self.start < other.start
        return self.start < other


@dataclass
class TranscriptData:
    """
    Full transcript with timing information.
    
    Uses binary search for optimized range queries.
    """
    segments: List[TranscriptSegment] = field(default_factory=list)
    speech_segments: List[Tuple[float, float]] = field(default_factory=list)
    
    # Index for binary search
    _segment_starts: List[float] = field(default_factory=list)
    _built: bool = False
    
    def build_index(self):
        """Build search index. Call after adding all segments."""
        self.segments.sort(key=lambda s: s.start)
        self._segment_starts = [s.start for s in self.segments]
        self._built = True
    
    def get_text_for_range(self, start: float, end: float) -> str:
        """
        Get transcript text within a time range.
        
        O(log n + k) where k = overlapping segments.
        """
        if not self._built:
            self.build_index()
        
        if not self.segments:
            return ""
        
        # Find first segment that might overlap
        first_idx = bisect.bisect_left(self._segment_starts, start)
        
        # Go back to catch segments that started earlier but extend into range
        while first_idx > 0 and self.segments[first_idx - 1].end > start:
            first_idx -= 1
        
        texts = []
        for i in range(first_idx, len(self.segments)):
            seg = self.segments[i]
            
            if seg.start >= end:
                break
            
            if seg.end > start:
                texts.append(seg.text)
        
        return " ".join(texts)
    
    def get_context_before(
        self, 
        timestamp: float, 
        sentences: int = 2
    ) -> str:
        """
        Get N sentences before a timestamp.
        
        O(log n + sentences).
        """
        if not self._built:
            self.build_index()
        
        if not self.segments:
            return ""
        
        # Find first segment at or after timestamp
        idx = bisect.bisect_left(self._segment_starts, timestamp)
        
        # Go back to find segments before
        texts = []
        for i in range(idx - 1, -1, -1):
            seg = self.segments[i]
            if seg.end <= timestamp:
                texts.insert(0, seg.text)
                if len(texts) >= sentences:
                    break
        
        return " ".join(texts)
    
    def get_context_after(
        self, 
        timestamp: float, 
        sentences: int = 2
    ) -> str:
        """
        Get N sentences after a timestamp.
        
        O(log n + sentences).
        """
        if not self._built:
            self.build_index()
        
        if not self.segments:
            return ""
        
        # Find first segment at or after timestamp
        idx = bisect.bisect_left(self._segment_starts, timestamp)
        
        texts = []
        for i in range(idx, len(self.segments)):
            seg = self.segments[i]
            if seg.start >= timestamp:
                texts.append(seg.text)
                if len(texts) >= sentences:
                    break
        
        return " ".join(texts)
    
    def get_word_count_in_range(self, start: float, end: float) -> int:
        """Get approximate word count in a range."""
        text = self.get_text_for_range(start, end)
        return len(text.split())
    
    def get_speech_density(self, timestamp: float, window: float = 2.0) -> float:
        """
        Get words per second around a timestamp.
        
        Useful for detecting risky cut points (high density = mid-speech).
        """
        text = self.get_text_for_range(timestamp - window/2, timestamp + window/2)
        word_count = len(text.split())
        return word_count / window if window > 0 else 0.0
