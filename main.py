#!/usr/bin/env python3
"""
Video Split Pipeline v2 - Production-grade video segmentation.

Usage:
    python main.py video.mp4 --mode coverage --output output/
    python main.py video.mp4 --mode highlights --max-clips 5
"""

import argparse
import logging
import sys
from pathlib import Path

from src.pipeline import VideoSplitPipeline, PipelineConfig
from src.graph.boundary_graph import StitchBehavior


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    
    # Reduce noise from external libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Video Split Pipeline v2 - Production-grade video segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process video with full coverage mode
    python main.py video.mp4 --mode coverage

    # Extract top 5 highlights
    python main.py video.mp4 --mode highlights --max-clips 5

    # Custom duration range
    python main.py video.mp4 --min-duration 30 --max-duration 45

    # Disable LLM review (faster, less accurate)
    python main.py video.mp4 --no-llm

    # With stitch marks
    python main.py video.mp4 --stitch-marks 60.5,120.0,180.5
        """,
    )
    
    # Required
    parser.add_argument(
        "video",
        type=Path,
        help="Input video file",
    )
    
    # Mode
    parser.add_argument(
        "--mode",
        choices=["coverage", "highlights"],
        default="coverage",
        help="Selection mode: 'coverage' for full video, 'highlights' for best clips (default: coverage)",
    )
    
    # Duration
    parser.add_argument(
        "--min-duration",
        type=float,
        default=45.0,
        help="Minimum clip duration in seconds (default: 45)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=60.0,
        help="Maximum clip duration in seconds (default: 60)",
    )
    
    # Highlights mode
    parser.add_argument(
        "--max-clips",
        type=int,
        default=10,
        help="Maximum number of clips in highlights mode (default: 10)",
    )
    
    # Coverage mode
    parser.add_argument(
        "--strict-coverage",
        action="store_true",
        help="Enforce strict coverage (minimal gaps)",
    )
    parser.add_argument(
        "--gap-penalty",
        type=float,
        default=0.5,
        help="Penalty per second of gap between clips (default: 0.5)",
    )
    parser.add_argument(
        "--max-gap",
        type=float,
        default=5.0,
        help="Maximum allowed gap between clips in seconds (default: 5.0)",
    )
    
    # Stitch marks
    parser.add_argument(
        "--stitch-marks",
        type=str,
        default=None,
        help="Comma-separated stitch mark timestamps (e.g., '60.5,120.0,180.5')",
    )
    parser.add_argument(
        "--stitch-behavior",
        choices=["forbidden", "preferred", "must-cut"],
        default="forbidden",
        help="How to treat stitch marks (default: forbidden)",
    )
    
    # Review
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM review (faster, uses heuristics only)",
    )
    parser.add_argument(
        "--max-review-iterations",
        type=int,
        default=3,
        help="Maximum review iterations per clip (default: 3)",
    )
    
    # Detectors
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable voice activity detection",
    )
    parser.add_argument(
        "--no-transcript",
        action="store_true",
        help="Disable transcription",
    )
    parser.add_argument(
        "--no-shot",
        action="store_true",
        help="Disable shot boundary detection",
    )
    parser.add_argument(
        "--use-speaker",
        action="store_true",
        help="Enable speaker diarization (slower)",
    )
    
    # Output
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output"),
        help="Output directory (default: output/)",
    )
    
    # Cache
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache"),
        help="Cache directory (default: .cache/)",
    )
    
    # Misc
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate input
    if not args.video.exists():
        logger.error(f"Video file not found: {args.video}")
        sys.exit(1)
    
    # Parse stitch marks
    stitch_marks = None
    if args.stitch_marks:
        try:
            stitch_marks = [float(x.strip()) for x in args.stitch_marks.split(",")]
            logger.info(f"Stitch marks: {stitch_marks}")
        except ValueError:
            logger.error(f"Invalid stitch marks format: {args.stitch_marks}")
            sys.exit(1)
    
    # Map stitch behavior
    stitch_behavior_map = {
        "forbidden": StitchBehavior.FORBIDDEN_INTERIOR,
        "preferred": StitchBehavior.PREFERRED_CUT,
        "must-cut": StitchBehavior.MUST_CUT,
    }
    stitch_behavior = stitch_behavior_map[args.stitch_behavior]
    
    # Create config
    config = PipelineConfig(
        mode=args.mode,
        min_clip_duration=args.min_duration,
        max_clip_duration=args.max_duration,
        max_clips=args.max_clips,
        gap_penalty=args.gap_penalty,
        max_gap=args.max_gap,
        strict_coverage=args.strict_coverage,
        stitch_behavior=stitch_behavior,
        use_llm_review=not args.no_llm,
        max_review_iterations=args.max_review_iterations,
        use_vad=not args.no_vad,
        use_transcript=not args.no_transcript,
        use_shot=not args.no_shot,
        use_speaker=args.use_speaker,
        output_dir=args.output,
        cache_dir=args.cache_dir,
        use_cache=not args.no_cache,
    )
    
    # Create pipeline
    pipeline = VideoSplitPipeline(config)
    
    # Process
    try:
        results = pipeline.process(args.video, stitch_marks=stitch_marks)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    approved = [r for r in results if r.passed_review]
    failed = [r for r in results if not r.passed_review]
    
    print(f"\nApproved: {len(approved)}/{len(results)} clips")
    
    if approved:
        print("\nApproved clips:")
        total_duration = 0
        for i, r in enumerate(approved):
            duration = r.clip.duration
            total_duration += duration
            print(f"  {i+1}. {r.clip.start:.1f}s - {r.clip.end:.1f}s ({duration:.1f}s)")
            if r.final_path:
                print(f"      -> {r.final_path}")
        print(f"\nTotal duration: {total_duration:.1f}s")
    
    if failed and args.verbose:
        print("\nFailed clips:")
        for i, r in enumerate(failed):
            print(f"  - {r.clip.start:.1f}s - {r.clip.end:.1f}s")
    
    print(f"\nOutput directory: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
