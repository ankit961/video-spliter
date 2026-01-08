#!/usr/bin/env python3
"""Integration tests for the video split pipeline."""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph.boundary_graph import Boundary, BoundaryGraph, BoundaryType, StitchBehavior
from src.graph.candidate_edges import generate_candidate_edges, ClipFeatureComputer
from src.graph.scorer import ClipScorer, ScoringWeights
from src.optimizer.dp_coverage import optimize_coverage, CoverageConfig
from src.optimizer.greedy_highlights import select_highlights, HighlightsConfig
from src.reviewer.heuristic_checks import run_heuristic_checks, HeuristicResult


def test_boundary_graph_basics():
    """Test basic boundary graph operations."""
    print("Test: boundary_graph_basics")
    
    graph = BoundaryGraph(video_duration=100.0)
    graph.add(Boundary(timestamp=50.0, types={BoundaryType.VAD_PAUSE}))
    graph.finalize()
    
    # Should have VIDEO_START at 0 and VIDEO_END at 100
    assert graph.boundaries[0].timestamp == 0.0
    assert BoundaryType.VIDEO_START in graph.boundaries[0].types
    assert graph.boundaries[-1].timestamp == 100.0
    assert BoundaryType.VIDEO_END in graph.boundaries[-1].types
    
    print("  ✓ Passed")


def test_anchor_preservation():
    """Test that anchor timestamps are preserved during merge."""
    print("Test: anchor_preservation")
    
    b1 = Boundary(
        timestamp=10.0,
        types={BoundaryType.VAD_PAUSE},
        confidences={BoundaryType.VAD_PAUSE: 0.9},
    )
    b2 = Boundary(
        timestamp=10.2,
        types={BoundaryType.STITCH_MARK},
        confidences={BoundaryType.STITCH_MARK: 1.0},
    )
    
    b1.merge_from(b2)
    
    # Should keep stitch mark timestamp (higher priority)
    assert b1.timestamp == 10.2
    assert BoundaryType.STITCH_MARK in b1.types
    assert BoundaryType.VAD_PAUSE in b1.types
    
    print("  ✓ Passed")


def test_stitch_interior_detection():
    """Test O(log m) stitch interior detection."""
    print("Test: stitch_interior_detection")
    
    graph = BoundaryGraph(video_duration=100.0)
    graph.add_stitch_mark(50.0)
    graph.finalize()
    
    # Clip 0-40 should NOT contain stitch
    assert not graph.has_stitch_interior(0.0, 40.0)
    
    # Clip 0-60 SHOULD contain stitch
    assert graph.has_stitch_interior(0.0, 60.0)
    
    # Clip starting AT stitch should NOT contain interior stitch
    assert not graph.has_stitch_interior(50.0, 100.0)
    
    # Clip ending AT stitch should NOT contain interior stitch
    assert not graph.has_stitch_interior(0.0, 50.0)
    
    print("  ✓ Passed")


def test_candidate_generation():
    """Test candidate clip generation."""
    print("Test: candidate_generation")
    
    graph = BoundaryGraph(video_duration=180.0)
    for t in range(0, 181, 5):
        graph.add(Boundary(timestamp=float(t), types={BoundaryType.VAD_PAUSE}))
    graph.finalize()
    
    candidates = list(generate_candidate_edges(graph, min_duration=45.0, max_duration=60.0))
    
    assert len(candidates) > 0
    
    for c in candidates:
        assert 45.0 <= c.duration <= 60.0
    
    print(f"  ✓ Passed ({len(candidates)} candidates)")


def test_dp_coverage():
    """Test DP coverage optimizer."""
    print("Test: dp_coverage")
    
    graph = BoundaryGraph(video_duration=180.0)
    for t in range(0, 181, 5):
        graph.add(Boundary(timestamp=float(t), types={BoundaryType.VAD_PAUSE}))
    graph.finalize()
    
    scorer = ClipScorer()
    config = CoverageConfig(
        max_gap_allowed=15.0,
        max_tail_gap=60.0,
    )
    
    result = optimize_coverage(graph, scorer, config)
    
    assert len(result.clips) > 0
    assert result.coverage_ratio > 0.9  # Should achieve good coverage
    
    # Verify no overlaps
    sorted_clips = sorted(result.clips, key=lambda c: c.start)
    for i in range(1, len(sorted_clips)):
        assert sorted_clips[i].start >= sorted_clips[i-1].end - 0.1  # Allow tiny overlap
    
    print(f"  ✓ Passed ({len(result.clips)} clips, {result.coverage_ratio:.1%} coverage)")


def test_stitch_constraint_enforcement():
    """Test that stitch constraints are properly enforced."""
    print("Test: stitch_constraint_enforcement")
    
    graph = BoundaryGraph(video_duration=300.0)
    for t in range(0, 301, 15):
        graph.add(Boundary(timestamp=float(t), types={BoundaryType.VAD_PAUSE}))
    graph.add_stitch_mark(150.0)
    graph.stitch_behavior = StitchBehavior.FORBIDDEN_INTERIOR
    graph.finalize()
    
    scorer = ClipScorer(stitch_behavior=StitchBehavior.FORBIDDEN_INTERIOR)
    config = CoverageConfig(max_gap_allowed=20.0, max_tail_gap=60.0)
    
    result = optimize_coverage(graph, scorer, config)
    
    # No selected clip should cross the stitch
    for clip in result.clips:
        assert not graph.has_stitch_interior(clip.start, clip.end), \
            f"Clip [{clip.start:.0f}s - {clip.end:.0f}s] crosses stitch!"
    
    print(f"  ✓ Passed (no clips cross stitch at 150s)")


def test_greedy_highlights():
    """Test greedy highlights selector."""
    print("Test: greedy_highlights")
    
    graph = BoundaryGraph(video_duration=180.0)
    for t in range(0, 181, 5):
        graph.add(Boundary(timestamp=float(t), types={BoundaryType.VAD_PAUSE}))
    graph.finalize()
    
    scorer = ClipScorer()
    config = HighlightsConfig(max_clips=3)
    
    highlights = select_highlights(graph, scorer, config)
    
    assert len(highlights) <= 3
    
    # Verify no overlaps
    for i, a in enumerate(highlights):
        for j, b in enumerate(highlights):
            if i < j:
                assert a.end <= b.start or b.end <= a.start, \
                    f"Clips overlap: [{a.start:.0f}-{a.end:.0f}] and [{b.start:.0f}-{b.end:.0f}]"
    
    print(f"  ✓ Passed ({len(highlights)} non-overlapping highlights)")


def test_heuristic_checks():
    """Test heuristic review checks."""
    print("Test: heuristic_checks")
    
    graph = BoundaryGraph(video_duration=180.0)
    for t in range(0, 181, 5):
        graph.add(Boundary(timestamp=float(t), types={BoundaryType.VAD_PAUSE}))
    graph.finalize()
    
    candidates = list(generate_candidate_edges(graph, 45.0, 60.0))
    clip = candidates[0]
    
    result = run_heuristic_checks(clip)
    
    assert result.result in [HeuristicResult.PASS, HeuristicResult.FAIL, HeuristicResult.NEEDS_LLM]
    
    print(f"  ✓ Passed (result: {result.result.value})")


def test_multi_signal_boundaries():
    """Test that multi-signal boundaries get bonus scoring."""
    print("Test: multi_signal_boundaries")
    
    graph = BoundaryGraph(video_duration=120.0)
    
    # Single signal boundary
    graph.add(Boundary(timestamp=0.0, types={BoundaryType.VAD_PAUSE}))
    
    # Multi-signal boundary
    graph.add(Boundary(
        timestamp=60.0, 
        types={BoundaryType.VAD_PAUSE, BoundaryType.SHOT, BoundaryType.SENTENCE_END}
    ))
    
    graph.add(Boundary(timestamp=120.0, types={BoundaryType.VAD_PAUSE}))
    graph.finalize()
    
    # Check multi-signal detection
    b_multi = [b for b in graph.boundaries if b.timestamp == 60.0][0]
    assert b_multi.is_multi_signal
    
    b_single = [b for b in graph.boundaries if b.timestamp == 0.0][0]
    assert not b_single.is_multi_signal
    
    print("  ✓ Passed")


def run_all_tests():
    """Run all integration tests."""
    print("=" * 50)
    print("Video Split Pipeline - Integration Tests")
    print("=" * 50)
    print()
    
    tests = [
        test_boundary_graph_basics,
        test_anchor_preservation,
        test_stitch_interior_detection,
        test_candidate_generation,
        test_dp_coverage,
        test_stitch_constraint_enforcement,
        test_greedy_highlights,
        test_heuristic_checks,
        test_multi_signal_boundaries,
    ]
    
    passed = 0
    failed = 0
    
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1
    
    print()
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
