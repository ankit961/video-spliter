"""Main pipeline orchestrator."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Literal
import logging
import hashlib

from .boundaries.detector import UnifiedBoundaryDetector, DetectorConfig
from .graph.boundary_graph import BoundaryGraph, StitchBehavior, BoundaryType
from .graph.candidate_edges import CandidateClip, ClipFeatureComputer
from .graph.scorer import ClipScorer, ScoringWeights
from .optimizer.dp_coverage import optimize_coverage, CoverageConfig
from .optimizer.greedy_highlights import select_highlights, HighlightsConfig
from .reviewer.heuristic_checks import run_heuristic_checks, HeuristicResult
from .reviewer.llm_reviewer import LLMReviewer, ReviewDecision
from .render.preview import render_preview
from .render.final import render_final
from .utils.cache import PipelineCache
from .utils.transcript import TranscriptData

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the video split pipeline."""
    # Mode
    mode: Literal["coverage", "highlights"] = "coverage"
    
    # Duration constraints
    min_clip_duration: float = 45.0
    max_clip_duration: float = 60.0
    
    # Coverage mode settings
    gap_penalty: float = 0.5
    max_gap: float = 5.0
    strict_coverage: bool = False
    enforce_tail_coverage: bool = True
    
    # Highlights mode settings
    max_clips: int = 10
    
    # Stitch behavior
    stitch_behavior: StitchBehavior = StitchBehavior.FORBIDDEN_INTERIOR
    
    # Review settings
    use_llm_review: bool = True
    max_review_iterations: int = 3
    
    # Render settings
    preview_resolution: str = "480p"
    render_preview_only_for_borderline: bool = True
    
    # Output
    output_dir: Path = Path("output")
    
    # Cache
    cache_dir: Path = Path(".cache")
    use_cache: bool = True
    
    # Detector settings
    use_vad: bool = True
    use_transcript: bool = True
    use_shot: bool = True
    use_speaker: bool = False


@dataclass
class ClipResult:
    """Result for a single processed clip."""
    clip: CandidateClip
    transcript: str
    passed_review: bool
    review_iterations: int
    was_borderline: bool = False
    final_path: Optional[Path] = None
    preview_path: Optional[Path] = None


class VideoSplitPipeline:
    """
    Main pipeline orchestrator.
    
    Flow:
    1. Extract boundaries (shot, VAD, transcript, stitch, energy)
    2. Build boundary graph
    3. Generate candidate clips
    4. Optimize selection (coverage or highlights)
    5. Review loop (heuristics → LLM if needed)
    6. Render approved clips
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
        # Initialize components
        detector_config = DetectorConfig(
            use_vad=self.config.use_vad,
            use_transcript=self.config.use_transcript,
            use_shot=self.config.use_shot,
            use_speaker=self.config.use_speaker,
        )
        self.boundary_detector = UnifiedBoundaryDetector(detector_config)
        self.scorer = ClipScorer(stitch_behavior=self.config.stitch_behavior)
        self.llm_reviewer = LLMReviewer() if self.config.use_llm_review else None
        self.cache = PipelineCache(
            self.config.cache_dir, 
            enabled=self.config.use_cache
        )
    
    def process(
        self, 
        video_path: Path,
        stitch_marks: Optional[List[float]] = None,
    ) -> List[ClipResult]:
        """
        Process a video and return approved clips.
        
        Args:
            video_path: Path to input video
            stitch_marks: Optional list of stitch mark timestamps
            
        Returns:
            List of ClipResult with approved clips
        """
        video_path = Path(video_path)
        logger.info(f"Processing: {video_path}")
        
        cache_key = self._compute_cache_key(video_path)
        
        # === Step 1: Build boundary graph ===
        logger.info("Building boundary graph...")
        graph = self.cache.get_or_compute(
            f"{cache_key}_graph",
            lambda: self._build_graph(video_path, stitch_marks),
        )
        logger.info(f"Graph: {len(graph.boundaries)} boundaries")
        
        # === Step 2: Get transcript ===
        logger.info("Loading transcript...")
        transcript_data: TranscriptData = self.cache.get_or_compute(
            f"{cache_key}_transcript",
            lambda: self._get_transcript(video_path),
        )
        transcript_data.build_index()
        
        # === Step 3: Create feature computer ===
        speech_segments = self.boundary_detector.get_speech_segments()
        if not speech_segments:
            speech_segments = transcript_data.speech_segments
        feature_computer = ClipFeatureComputer(graph, speech_segments)
        
        # === Step 4: Select clips ===
        logger.info(f"Selecting clips (mode={self.config.mode})...")
        clips = self._select_clips(graph, speech_segments)
        logger.info(f"Selected {len(clips)} candidate clips")
        
        if not clips:
            logger.warning("No valid clips found!")
            return []
        
        # === Step 5: Review loop ===
        logger.info("Running review loop...")
        results = []
        
        for i, clip in enumerate(clips):
            logger.info(f"Reviewing clip {i+1}/{len(clips)}: {clip.start:.1f}s - {clip.end:.1f}s")
            
            result = self._review_clip(
                clip,
                graph,
                transcript_data,
                feature_computer,
            )
            results.append(result)
            
            status = "✓ PASS" if result.passed_review else "✗ FAIL"
            logger.info(f"  {status} (iterations: {result.review_iterations})")
        
        # === Step 6: Render ===
        logger.info("Rendering clips...")
        self._render_results(video_path, results)
        
        # Summary
        approved = sum(1 for r in results if r.passed_review)
        logger.info(f"Done! {approved}/{len(results)} clips approved")
        
        return results
    
    def _build_graph(
        self, 
        video_path: Path,
        stitch_marks: Optional[List[float]] = None,
    ) -> BoundaryGraph:
        """Build and finalize boundary graph."""
        graph = self.boundary_detector.detect_all(video_path)
        
        # Add stitch marks if provided
        if stitch_marks:
            for mark in stitch_marks:
                graph.add_stitch_mark(mark)
        
        graph.stitch_behavior = self.config.stitch_behavior
        
        # Finalize handles: sort, inject endpoints, merge, downsample
        graph.finalize(
            merge_threshold=0.3,
            downsample_sentence_gap=0.5,
        )
        
        return graph
    
    def _get_transcript(self, video_path: Path) -> TranscriptData:
        """Get transcript data."""
        return self.boundary_detector.transcribe(video_path)
    
    def _select_clips(
        self,
        graph: BoundaryGraph,
        speech_segments: List,
    ) -> List[CandidateClip]:
        """Select clips based on mode."""
        if self.config.mode == "coverage":
            opt_config = CoverageConfig(
                min_duration=self.config.min_clip_duration,
                max_duration=self.config.max_clip_duration,
                gap_penalty_per_second=self.config.gap_penalty,
                max_gap_allowed=self.config.max_gap,
                strict_coverage=self.config.strict_coverage,
                enforce_tail_coverage=self.config.enforce_tail_coverage,
            )
            result = optimize_coverage(graph, self.scorer, opt_config, speech_segments)
            logger.info(
                f"Coverage: {result.coverage_ratio:.1%}, "
                f"tail_gap: {result.tail_gap:.1f}s, "
                f"total_gap: {result.total_gap_seconds:.1f}s"
            )
            return result.clips
        else:
            hl_config = HighlightsConfig(
                min_duration=self.config.min_clip_duration,
                max_duration=self.config.max_clip_duration,
                max_clips=self.config.max_clips,
            )
            return select_highlights(graph, self.scorer, hl_config, speech_segments)
    
    def _review_clip(
        self,
        clip: CandidateClip,
        graph: BoundaryGraph,
        transcript_data: TranscriptData,
        feature_computer: ClipFeatureComputer,
    ) -> ClipResult:
        """Run review loop on a single clip."""
        current_clip = clip
        iterations = 0
        was_borderline = False
        
        for iteration in range(self.config.max_review_iterations):
            iterations = iteration + 1
            
            # Get fresh transcript for current clip bounds
            clip_transcript = transcript_data.get_text_for_range(
                current_clip.start, current_clip.end
            )
            
            # Heuristic checks
            heuristic_result = run_heuristic_checks(current_clip)
            
            if heuristic_result.result == HeuristicResult.PASS:
                return ClipResult(
                    clip=current_clip,
                    transcript=clip_transcript,
                    passed_review=True,
                    review_iterations=iterations,
                    was_borderline=was_borderline,
                )
            
            if heuristic_result.result == HeuristicResult.FAIL:
                logger.debug(f"  Heuristic FAIL: {heuristic_result.failed_checks}")
                return ClipResult(
                    clip=current_clip,
                    transcript=clip_transcript,
                    passed_review=False,
                    review_iterations=iterations,
                    was_borderline=was_borderline,
                )
            
            # Borderline — needs LLM
            was_borderline = True
            logger.debug(f"  Borderline: {heuristic_result.failed_checks}")
            
            if not self.llm_reviewer:
                # No LLM, accept borderline as pass
                return ClipResult(
                    clip=current_clip,
                    transcript=clip_transcript,
                    passed_review=True,
                    review_iterations=iterations,
                    was_borderline=True,
                )
            
            # Get context for LLM
            context_before = transcript_data.get_context_before(current_clip.start)
            context_after = transcript_data.get_context_after(current_clip.end)
            
            # LLM review
            review = self.llm_reviewer.review(
                current_clip,
                clip_transcript,
                graph,
                context_before=context_before,
                context_after=context_after,
            )
            
            logger.debug(f"  LLM: {review.decision.value}, reason: {review.reason_code}")
            
            if review.decision == ReviewDecision.PASS:
                return ClipResult(
                    clip=current_clip,
                    transcript=clip_transcript,
                    passed_review=True,
                    review_iterations=iterations,
                    was_borderline=True,
                )
            
            # Apply shifts and retry
            shifted = self.llm_reviewer.apply_shifts(
                current_clip,
                review,
                graph,
                feature_computer,
            )
            
            if shifted is None:
                logger.debug("  Invalid shift suggested")
                return ClipResult(
                    clip=current_clip,
                    transcript=clip_transcript,
                    passed_review=False,
                    review_iterations=iterations,
                    was_borderline=True,
                )
            
            logger.debug(f"  Shifted: start={review.start_shift.value}, end={review.end_shift.value}")
            current_clip = shifted
        
        # Exhausted iterations
        return ClipResult(
            clip=current_clip,
            transcript=transcript_data.get_text_for_range(
                current_clip.start, current_clip.end
            ),
            passed_review=False,
            review_iterations=iterations,
            was_borderline=True,
        )
    
    def _render_results(self, video_path: Path, results: List[ClipResult]):
        """Render approved clips."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, result in enumerate(results):
            if not result.passed_review:
                continue
            
            # Preview (only for borderline if configured)
            if self.config.render_preview_only_for_borderline:
                if result.was_borderline:
                    preview_path = self.config.output_dir / f"preview_{i:03d}.mp4"
                    render_preview(
                        video_path,
                        result.clip.start,
                        result.clip.end,
                        preview_path,
                        resolution=self.config.preview_resolution,
                    )
                    result.preview_path = preview_path
            else:
                preview_path = self.config.output_dir / f"preview_{i:03d}.mp4"
                render_preview(
                    video_path,
                    result.clip.start,
                    result.clip.end,
                    preview_path,
                    resolution=self.config.preview_resolution,
                )
                result.preview_path = preview_path
            
            # Final render
            final_path = self.config.output_dir / f"clip_{i:03d}.mp4"
            render_final(
                video_path,
                result.clip.start,
                result.clip.end,
                final_path,
            )
            result.final_path = final_path
    
    def _compute_cache_key(self, video_path: Path) -> str:
        """Compute cache key from video + config."""
        # File hash (first 1MB for speed)
        with open(video_path, 'rb') as f:
            file_hash = hashlib.md5(f.read(1024 * 1024)).hexdigest()[:12]
        
        # Config hash
        config_str = (
            f"{self.config.min_clip_duration}_"
            f"{self.config.max_clip_duration}_"
            f"{self.config.stitch_behavior.value}_"
            f"{self.config.mode}_"
            f"{self.config.use_vad}_{self.config.use_transcript}_{self.config.use_shot}"
        )
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        return f"{video_path.stem}_{file_hash}_{config_hash}"
