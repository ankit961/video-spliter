"""Tests for boundary graph."""

import pytest
from src.graph.boundary_graph import (
    Boundary, 
    BoundaryGraph, 
    BoundaryType, 
    StitchBehavior,
)


class TestBoundary:
    """Tests for Boundary class."""
    
    def test_create_boundary(self):
        b = Boundary(
            timestamp=10.0,
            types={BoundaryType.VAD_PAUSE},
            confidences={BoundaryType.VAD_PAUSE: 0.9},
        )
        assert b.timestamp == 10.0
        assert b.confidence == 0.9
        assert b.has_type(BoundaryType.VAD_PAUSE)
    
    def test_multi_signal(self):
        b = Boundary(
            timestamp=10.0,
            types={BoundaryType.VAD_PAUSE, BoundaryType.SENTENCE_END},
        )
        assert b.is_multi_signal
    
    def test_merge_preserves_anchor(self):
        """Anchor timestamps should be preserved during merge."""
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


class TestBoundaryGraph:
    """Tests for BoundaryGraph class."""
    
    def test_finalize_adds_endpoints(self):
        graph = BoundaryGraph(video_duration=100.0)
        graph.add(Boundary(timestamp=50.0, types={BoundaryType.VAD_PAUSE}))
        graph.finalize()
        
        # Should have VIDEO_START at 0 and VIDEO_END at 100
        assert graph.boundaries[0].timestamp == 0.0
        assert BoundaryType.VIDEO_START in graph.boundaries[0].types
        
        assert graph.boundaries[-1].timestamp == 100.0
        assert BoundaryType.VIDEO_END in graph.boundaries[-1].types
    
    def test_stitch_interior_detection(self):
        graph = BoundaryGraph(video_duration=100.0)
        graph.add_stitch_mark(50.0)
        graph.finalize()
        
        # Clip 0-40 should NOT contain stitch
        assert not graph.has_stitch_interior(0.0, 40.0)
        
        # Clip 0-60 SHOULD contain stitch
        assert graph.has_stitch_interior(0.0, 60.0)
        
        # Clip starting AT stitch should NOT contain interior stitch
        assert not graph.has_stitch_interior(50.0, 100.0)
    
    def test_merge_nearby(self):
        graph = BoundaryGraph(video_duration=100.0)
        graph.add(Boundary(timestamp=10.0, types={BoundaryType.VAD_PAUSE}))
        graph.add(Boundary(timestamp=10.1, types={BoundaryType.SENTENCE_END}))
        graph.add(Boundary(timestamp=50.0, types={BoundaryType.SHOT}))
        
        graph.finalize(merge_threshold=0.3)
        
        # 10.0 and 10.1 should be merged
        mid_boundary = [b for b in graph.boundaries if 9.0 < b.timestamp < 11.0]
        assert len(mid_boundary) == 1
        assert mid_boundary[0].is_multi_signal


class TestCandidateGeneration:
    """Tests for candidate edge generation."""
    
    def test_generates_valid_candidates(self):
        from src.graph.candidate_edges import generate_candidate_edges
        
        graph = BoundaryGraph(video_duration=120.0)
        
        # Add boundaries
        for t in [0, 10, 20, 30, 45, 55, 60, 75, 90, 105, 120]:
            graph.add(Boundary(timestamp=float(t), types={BoundaryType.VAD_PAUSE}))
        
        graph.finalize()
        
        candidates = list(generate_candidate_edges(
            graph,
            min_duration=45.0,
            max_duration=60.0,
        ))
        
        # Should have some valid candidates
        assert len(candidates) > 0
        
        # All candidates should be within duration bounds
        for c in candidates:
            assert 45.0 <= c.duration <= 60.0
    
    def test_must_cut_filters_correctly(self):
        from src.graph.candidate_edges import generate_candidate_edges
        
        graph = BoundaryGraph(video_duration=120.0)
        graph.stitch_behavior = StitchBehavior.MUST_CUT
        
        # Add stitch marks
        graph.add_stitch_mark(0.0)
        graph.add_stitch_mark(55.0)
        graph.add_stitch_mark(120.0)
        
        # Add some non-stitch boundaries
        for t in [10, 20, 30, 45]:
            graph.add(Boundary(timestamp=float(t), types={BoundaryType.VAD_PAUSE}))
        
        graph.finalize()
        
        candidates = list(generate_candidate_edges(
            graph,
            min_duration=45.0,
            max_duration=60.0,
        ))
        
        # All candidates should start/end at stitch or video boundaries
        for c in candidates:
            start_ok = (
                c.start_boundary.has_type(BoundaryType.STITCH_MARK) or
                c.start_boundary.has_type(BoundaryType.VIDEO_START)
            )
            end_ok = (
                c.end_boundary.has_type(BoundaryType.STITCH_MARK) or
                c.end_boundary.has_type(BoundaryType.VIDEO_END)
            )
            assert start_ok, f"Invalid start: {c.start_boundary.types}"
            assert end_ok, f"Invalid end: {c.end_boundary.types}"


class TestOptimizer:
    """Tests for optimization."""
    
    def test_dp_coverage(self):
        from src.graph.candidate_edges import generate_candidate_edges
        from src.graph.scorer import ClipScorer
        from src.optimizer.dp_coverage import optimize_coverage, CoverageConfig
        
        graph = BoundaryGraph(video_duration=180.0)
        
        # Add boundaries every 10 seconds
        for t in range(0, 181, 10):
            graph.add(Boundary(timestamp=float(t), types={BoundaryType.VAD_PAUSE}))
        
        graph.finalize()
        
        scorer = ClipScorer()
        config = CoverageConfig(
            min_duration=45.0,
            max_duration=60.0,
        )
        
        result = optimize_coverage(graph, scorer, config)
        
        # Should have clips
        assert len(result.clips) > 0
        
        # Should have reasonable coverage
        assert result.coverage_ratio > 0.5
    
    def test_greedy_highlights(self):
        from src.graph.scorer import ClipScorer
        from src.optimizer.greedy_highlights import select_highlights, HighlightsConfig
        
        graph = BoundaryGraph(video_duration=300.0)
        
        # Add boundaries
        for t in range(0, 301, 10):
            graph.add(Boundary(timestamp=float(t), types={BoundaryType.VAD_PAUSE}))
        
        graph.finalize()
        
        scorer = ClipScorer()
        config = HighlightsConfig(
            min_duration=45.0,
            max_duration=60.0,
            max_clips=3,
        )
        
        clips = select_highlights(graph, scorer, config)
        
        # Should have at most max_clips
        assert len(clips) <= 3
        
        # Clips should not overlap
        for i in range(len(clips) - 1):
            assert clips[i].end <= clips[i + 1].start


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
