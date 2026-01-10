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
    base_clip_score: float = 10.0  # Increased from 8.0 to prioritize coverage
    
    # Boundary quality by type
    boundary_type_weights: Dict[BoundaryType, float] = None
    
    # Multi-signal bonus (boundary detected by multiple methods)
    multi_signal_bonus: float = 0.5  # Increased for cleaner cuts
    
    # Shot + energy alignment bonus (ideal cut point)
    shot_energy_alignment_bonus: float = 0.8
    
    # Duration preferences
    target_duration: float = 55.0
    duration_penalty_per_second: float = 0.03  # Reduced
    hard_duration_penalty: float = 1.0  # Reduced
    
    # Content quality
    hook_bonus: float = 0.5
    silence_penalty_multiplier: float = 1.0  # Reduced
    high_speech_density_penalty: float = 0.5  # Reduced
    hard_stitch_penalty: float = 10.0
    
    # Stitch alignment (for PREFERRED_CUT mode)
    stitch_alignment_bonus: float = 0.5
    
    # Transcript-aware scoring
    sentence_alignment_bonus: float = 1.5  # Bonus for sentence-aligned cuts
    mid_sentence_penalty: float = 0.8      # Reduced penalty for cutting mid-sentence
    
    # Audio-abrupt cut penalty (visual-only cuts are jarring for music content)
    audio_abrupt_penalty: float = 0.5  # Reduced penalty for cuts with no audio signal
    
    # Reach-aware bonuses (prefer clips that push coverage forward)
    tail_reach_bonus_far: float = 2.0      # Clips reaching >80% of video
    tail_reach_bonus_mid: float = 1.0      # Clips reaching 60-80% of video
    
    def __post_init__(self):
        if self.boundary_type_weights is None:
            self.boundary_type_weights = {
                BoundaryType.SHOT: 1.0,
                BoundaryType.STITCH_MARK: 0.95,
                BoundaryType.VAD_PAUSE: 0.95,    # Increased from 0.9: pauses are better than shots alone
                BoundaryType.SPEAKER_TURN: 0.85,
                BoundaryType.SENTENCE_END: 1.1,  # Sentence ends are ideal cuts (but rare in singing)
                BoundaryType.ENERGY_DIP: 1.05,   # Increased from 0.65: music content needs energy signals
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
        
        # === Audio-abrupt cut detection ===
        # For music/kids content, cuts with no audio signal are jarring
        audio_signals = {BoundaryType.VAD_PAUSE, BoundaryType.ENERGY_DIP, 
                        BoundaryType.SENTENCE_END, BoundaryType.SPEAKER_TURN}
        
        start_has_audio = bool(clip.start_boundary.types & audio_signals)
        end_has_audio = bool(clip.end_boundary.types & audio_signals)
        
        # Penalize visual-only cuts (especially at end where audio abruptly stops)
        if not end_has_audio:
            score -= w.audio_abrupt_penalty
        if not start_has_audio:
            score -= w.audio_abrupt_penalty * 0.5  # Start is less critical
        
        # === Transcript-aware scoring (critical for clean cuts) ===
        start_has_sentence = BoundaryType.SENTENCE_END in clip.start_boundary.types
        end_has_sentence = BoundaryType.SENTENCE_END in clip.end_boundary.types
        
        # Big bonus for sentence-aligned cuts (both start and end)
        if start_has_sentence and end_has_sentence:
            score += w.sentence_alignment_bonus
        elif end_has_sentence:
            # End alignment is more important (avoid cutting mid-sentence)
            score += w.sentence_alignment_bonus * 0.6
        elif start_has_sentence:
            score += w.sentence_alignment_bonus * 0.3
        
        # Penalty for high speech density at boundaries (likely mid-sentence cut)
        start_density = getattr(clip.start_boundary, 'speech_density', 0.0) or 0.0
        end_density = getattr(clip.end_boundary, 'speech_density', 0.0) or 0.0
        
        # Penalize cuts during rapid speech (mid-sentence indicator)
        if end_density > 2.5 and not end_has_sentence:
            score -= w.mid_sentence_penalty
        if start_density > 2.5 and not start_has_sentence:
            score -= w.mid_sentence_penalty * 0.5
        
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
        
        # === Speech density penalty (overall) ===
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
        
        # Check boundary types for alignment bonuses
        has_shot = BoundaryType.SHOT in boundary.types
        has_energy = BoundaryType.ENERGY_DIP in boundary.types
        has_vad_pause = BoundaryType.VAD_PAUSE in boundary.types
        has_sentence = BoundaryType.SENTENCE_END in boundary.types
        
        # === Triple alignment bonus (shot + sentence + pause/energy) ===
        # This is the ideal cut point for kids/music content
        if has_shot and has_sentence and (has_vad_pause or has_energy):
            best_score += w.shot_energy_alignment_bonus * 1.5
        # === Shot + sentence alignment (very clean cut) ===
        elif has_shot and has_sentence:
            best_score += w.shot_energy_alignment_bonus * 1.2
        # === Sentence + pause (clean audio cut) ===
        elif has_sentence and has_vad_pause:
            best_score += w.shot_energy_alignment_bonus * 1.0
        # === Shot + energy (visual transition with energy dip - good for music) ===
        elif has_shot and has_energy:
            best_score += w.shot_energy_alignment_bonus * 1.1  # Increased from 1.0
        # === Energy + pause (audio-only cut - good for singing) ===
        elif has_energy and has_vad_pause:
            best_score += w.shot_energy_alignment_bonus * 0.9  # Increased from 0.5
        # === Shot alone (less ideal than audio signals) ===
        elif has_shot and not has_vad_pause and not has_energy and not has_sentence:
            best_score += 0.3  # Penalize visual-only cuts in singing sections
        # === Energy dip alone (good for music/singing) ===
        elif has_energy and not has_shot:
            best_score += w.shot_energy_alignment_bonus * 0.8
        
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
