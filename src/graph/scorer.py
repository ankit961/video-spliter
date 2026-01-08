"""Clip scoring for candidate selection."""

from dataclasses import dataclass
from typing import Optional, Dict
import pickle
from pathlib import Path

from .candidate_edges import CandidateClip
from .boundary_graph import Boundary, BoundaryType, StitchBehavior


@dataclass
class ScoringWeights:
    """Tunable weights for clip scoring."""
    # Base score (ensures clips have positive value for coverage)
    base_clip_score: float = 5.0
    
    # Boundary quality by type
    boundary_type_weights: Dict[BoundaryType, float] = None
    
    # Multi-signal bonus (boundary detected by multiple methods)
    multi_signal_bonus: float = 0.3
    
    # Duration preferences
    target_duration: float = 55.0
    duration_penalty_per_second: float = 0.05
    hard_duration_penalty: float = 2.0
    
    # Content quality
    hook_bonus: float = 0.5
    silence_penalty_multiplier: float = 2.0
    high_speech_density_penalty: float = 1.0
    hard_stitch_penalty: float = 10.0
    
    # Stitch alignment (for PREFERRED_CUT mode)
    stitch_alignment_bonus: float = 0.5
    
    def __post_init__(self):
        if self.boundary_type_weights is None:
            self.boundary_type_weights = {
                BoundaryType.SHOT: 1.0,
                BoundaryType.STITCH_MARK: 0.95,
                BoundaryType.VAD_PAUSE: 0.9,
                BoundaryType.SPEAKER_TURN: 0.85,
                BoundaryType.SENTENCE_END: 0.7,
                BoundaryType.ENERGY_DIP: 0.5,
                BoundaryType.VIDEO_START: 0.8,
                BoundaryType.VIDEO_END: 0.8,
            }


class ClipScorer:
    """
    Scores candidate clips.
    
    Can use pure heuristics or a learned model.
    """
    
    def __init__(
        self, 
        weights: Optional[ScoringWeights] = None,
        stitch_behavior: StitchBehavior = StitchBehavior.FORBIDDEN_INTERIOR,
        learned_model_path: Optional[Path] = None,
    ):
        self.weights = weights or ScoringWeights()
        self.stitch_behavior = stitch_behavior
        self.learned_model = None
        
        if learned_model_path and learned_model_path.exists():
            with open(learned_model_path, 'rb') as f:
                self.learned_model = pickle.load(f)
    
    def score(self, clip: CandidateClip) -> float:
        """
        Score a candidate clip. Higher = better.
        """
        if self.learned_model:
            return self._score_learned(clip)
        return self._score_heuristic(clip)
    
    def _score_heuristic(self, clip: CandidateClip) -> float:
        """Score using hand-tuned heuristics."""
        w = self.weights
        score = w.base_clip_score  # Start with base score
        
        # === Boundary quality ===
        start_quality = self._boundary_quality(clip.start_boundary)
        end_quality = self._boundary_quality(clip.end_boundary)
        score += start_quality + end_quality
        
        # === Duration shaping ===
        duration_diff = abs(clip.duration - w.target_duration)
        if clip.duration < 45 or clip.duration > 60:
            score -= w.hard_duration_penalty
        else:
            score -= duration_diff * w.duration_penalty_per_second
        
        # === Hook bonus ===
        if clip.has_speech_in_first_500ms:
            score += w.hook_bonus
        
        # === Silence penalty ===
        if clip.silence_ratio > 0.15:
            score -= clip.silence_ratio * w.silence_penalty_multiplier
        
        # === Speech density penalty ===
        if clip.max_speech_density_at_boundaries > 3.0:
            score -= w.high_speech_density_penalty
        
        # === Stitch handling (based on behavior mode) ===
        if self.stitch_behavior == StitchBehavior.FORBIDDEN_INTERIOR:
            if clip.contains_forbidden_stitch:
                score -= w.hard_stitch_penalty
        
        elif self.stitch_behavior == StitchBehavior.PREFERRED_CUT:
            # Bonus for aligning to stitch marks
            if clip.starts_at_stitch:
                score += w.stitch_alignment_bonus
            if clip.ends_at_stitch:
                score += w.stitch_alignment_bonus
        
        return score
    
    def _boundary_quality(self, boundary: Boundary) -> float:
        """Get quality score for a boundary."""
        w = self.weights
        
        # Base: best type weight * confidence
        best_score = 0.0
        for btype in boundary.types:
            type_weight = w.boundary_type_weights.get(btype, 0.5)
            conf = boundary.confidences.get(btype, 1.0)
            best_score = max(best_score, type_weight * conf)
        
        # Multi-signal bonus
        if boundary.is_multi_signal:
            best_score += w.multi_signal_bonus
        
        return best_score
    
    def _score_learned(self, clip: CandidateClip) -> float:
        """Use learned model for scoring."""
        import math
        
        features = [
            self._boundary_quality(clip.start_boundary),
            self._boundary_quality(clip.end_boundary),
            clip.duration,
            float(clip.has_speech_in_first_500ms),
            clip.silence_ratio,
            clip.max_speech_density_at_boundaries,
            float(clip.contains_forbidden_stitch),
            float(clip.start_boundary.is_multi_signal),
            float(clip.end_boundary.is_multi_signal),
            clip.start_boundary.pause_duration or 0.0,
            clip.end_boundary.pause_duration or 0.0,
        ]
        
        prob = self.learned_model.predict_proba([features])[0][1]
        
        if prob <= 0.01:
            return -10.0
        if prob >= 0.99:
            return 10.0
        return math.log(prob / (1 - prob))
