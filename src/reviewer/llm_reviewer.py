"""LLM-based clip reviewer with structured outputs."""

from dataclasses import dataclass
from typing import Optional
from enum import Enum
import json
import logging

from ..graph.boundary_graph import BoundaryGraph
from ..graph.candidate_edges import CandidateClip, ClipFeatureComputer

logger = logging.getLogger(__name__)


class ReviewDecision(Enum):
    """LLM review decision."""
    PASS = "pass"
    FAIL = "fail"


class BoundaryShift(Enum):
    """Suggested boundary adjustment."""
    KEEP = "keep"
    PREV = "prev"
    NEXT = "next"


@dataclass
class ReviewResult:
    """Result from LLM review."""
    decision: ReviewDecision
    start_shift: BoundaryShift
    end_shift: BoundaryShift
    reason_code: str
    confidence: float


@dataclass
class ReviewerConfig:
    """Configuration for LLM reviewer."""
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_retries: int = 2
    context_sentences: int = 2


# JSON Schema for structured output
REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["pass", "fail"]},
        "start_shift": {"type": "string", "enum": ["keep", "prev", "next"]},
        "end_shift": {"type": "string", "enum": ["keep", "prev", "next"]},
        "reason_code": {
            "type": "string",
            "enum": [
                "good",
                "cut_mid_word",
                "weak_hook",
                "long_silence",
                "awkward_ending",
                "mid_thought",
                "low_energy",
                "other"
            ]
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
    },
    "required": ["decision", "start_shift", "end_shift", "reason_code", "confidence"],
    "additionalProperties": False
}


class LLMReviewer:
    """
    LLM-based clip reviewer with structured outputs.
    
    The LLM can ONLY:
    - Return PASS or FAIL
    - Suggest shifting start/end to adjacent boundaries
    - Provide a reason code
    
    It CANNOT suggest arbitrary timestamps.
    """
    
    def __init__(self, config: Optional[ReviewerConfig] = None):
        self.config = config or ReviewerConfig()
        self._client = None
    
    @property
    def client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        return self._client
    
    def review(
        self,
        clip: CandidateClip,
        transcript_segment: str,
        graph: BoundaryGraph,
        context_before: str = "",
        context_after: str = "",
    ) -> ReviewResult:
        """
        Review a single clip.
        
        Args:
            clip: The candidate clip to review
            transcript_segment: Transcript text for this clip
            graph: Boundary graph (to describe adjacent boundaries)
            context_before: Text before the clip
            context_after: Text after the clip
            
        Returns:
            ReviewResult with decision and suggested adjustments
        """
        prev_start = self._get_adjacent_boundary(graph, clip.start_idx, -1)
        next_start = self._get_adjacent_boundary(graph, clip.start_idx, +1)
        prev_end = self._get_adjacent_boundary(graph, clip.end_idx, -1)
        next_end = self._get_adjacent_boundary(graph, clip.end_idx, +1)
        
        system_prompt = """You are a video clip quality reviewer.
Your job is to evaluate if a clip has clean start and end points.

You can ONLY respond with the structured format. You cannot suggest custom timestamps.

Criteria for PASS:
- Starts with speech within 0.5s (good hook)
- No word is cut off at start or end
- No awkward silence at the end (>1s)
- Natural stopping point (not mid-thought)
- The clip makes sense without the context

If the clip fails, suggest shifting start or end to an adjacent boundary."""

        # Build transcript with context
        transcript_with_context = ""
        if context_before:
            transcript_with_context += f"[BEFORE CLIP]: {context_before}\n\n"
        transcript_with_context += f"[CLIP CONTENT]: {transcript_segment}"
        if context_after:
            transcript_with_context += f"\n\n[AFTER CLIP]: {context_after}"

        user_prompt = f"""Review this clip:

Duration: {clip.duration:.1f}s
Start: {clip.start:.2f}s ({self._format_boundary_types(clip.start_boundary)})
End: {clip.end:.2f}s ({self._format_boundary_types(clip.end_boundary)})

Transcript:
\"\"\"
{transcript_with_context}
\"\"\"

Adjacent boundaries:
- Previous start option: {prev_start}
- Next start option: {next_start}  
- Previous end option: {prev_end}
- Next end option: {next_end}"""

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                temperature=self.config.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "clip_review",
                        "strict": True,
                        "schema": REVIEW_SCHEMA,
                    }
                },
            )
            
            result = json.loads(response.choices[0].message.content)
            
            return ReviewResult(
                decision=ReviewDecision(result["decision"]),
                start_shift=BoundaryShift(result["start_shift"]),
                end_shift=BoundaryShift(result["end_shift"]),
                reason_code=result["reason_code"],
                confidence=result["confidence"],
            )
            
        except Exception as e:
            logger.error(f"LLM review failed: {e}")
            # Default to FAIL on error
            return ReviewResult(
                decision=ReviewDecision.FAIL,
                start_shift=BoundaryShift.KEEP,
                end_shift=BoundaryShift.KEEP,
                reason_code="other",
                confidence=0.0,
            )
    
    def _format_boundary_types(self, boundary) -> str:
        """Format boundary types for prompt."""
        types = [t.value for t in boundary.types]
        return ", ".join(types)
    
    def _get_adjacent_boundary(
        self, 
        graph: BoundaryGraph, 
        idx: int, 
        offset: int
    ) -> str:
        """Get description of adjacent boundary."""
        new_idx = idx + offset
        if new_idx < 0 or new_idx >= len(graph.boundaries):
            return "N/A (out of bounds)"
        
        b = graph.boundaries[new_idx]
        types = ", ".join(t.value for t in b.types)
        return f"{b.timestamp:.2f}s ({types}, conf={b.confidence:.2f})"
    
    def apply_shifts(
        self,
        clip: CandidateClip,
        review: ReviewResult,
        graph: BoundaryGraph,
        feature_computer: ClipFeatureComputer,
    ) -> Optional[CandidateClip]:
        """
        Apply boundary shifts to create adjusted clip.
        
        Recomputes all features via feature_computer.
        
        Returns None if shifts are invalid.
        """
        new_start_idx = clip.start_idx
        new_end_idx = clip.end_idx
        
        if review.start_shift == BoundaryShift.PREV:
            new_start_idx -= 1
        elif review.start_shift == BoundaryShift.NEXT:
            new_start_idx += 1
        
        if review.end_shift == BoundaryShift.PREV:
            new_end_idx -= 1
        elif review.end_shift == BoundaryShift.NEXT:
            new_end_idx += 1
        
        # Validate indices
        if new_start_idx < 0 or new_end_idx >= len(graph.boundaries):
            logger.warning("Shift would go out of bounds")
            return None
        if new_start_idx >= new_end_idx:
            logger.warning("Shift would create invalid clip (start >= end)")
            return None
        
        # Check duration bounds (relaxed for shifts)
        new_start = graph.boundaries[new_start_idx].timestamp
        new_end = graph.boundaries[new_end_idx].timestamp
        new_duration = new_end - new_start
        
        if new_duration < 30 or new_duration > 75:
            logger.warning(f"Shifted clip duration {new_duration:.1f}s out of bounds")
            return None
        
        # Recompute all features
        return feature_computer.compute(new_start_idx, new_end_idx)
