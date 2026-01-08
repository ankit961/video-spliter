"""Transcription using faster-whisper."""

from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
import re
import logging

from ..graph.boundary_graph import Boundary, BoundaryType, BoundaryGraph
from ..utils.transcript import TranscriptData, TranscriptSegment

logger = logging.getLogger(__name__)


@dataclass
class WhisperConfig:
    """Configuration for Whisper transcription."""
    model_size: str = "base"  # tiny, base, small, medium, large-v3
    language: Optional[str] = None  # None = auto-detect
    device: str = "auto"  # "auto", "cuda", "cpu"
    compute_type: str = "float16"  # float16, int8
    
    # Sentence boundary detection
    min_pause_for_boundary: float = 0.3
    sentence_end_patterns: List[str] = None
    
    def __post_init__(self):
        if self.sentence_end_patterns is None:
            self.sentence_end_patterns = [
                r'[.!?]$',
                r'[.!?]["\')\]]$',
            ]


class WhisperTranscriptDetector:
    """
    Transcription using faster-whisper.
    
    Produces:
    - TranscriptData with timed segments
    - SENTENCE_END boundaries
    - Speech density annotations
    """
    
    def __init__(self, config: Optional[WhisperConfig] = None):
        self.config = config or WhisperConfig()
        self._model = None
    
    def _load_model(self):
        """Lazy load faster-whisper model."""
        if self._model is not None:
            return
        
        from faster_whisper import WhisperModel
        
        device = self.config.device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        compute_type = self.config.compute_type if device == "cuda" else "int8"
        
        logger.info(f"Loading Whisper model: {self.config.model_size} on {device}")
        
        self._model = WhisperModel(
            self.config.model_size,
            device=device,
            compute_type=compute_type,
        )
    
    def transcribe(
        self,
        audio_path: Path,
        graph: BoundaryGraph,
    ) -> TranscriptData:
        """
        Transcribe audio and detect sentence boundaries.
        
        Args:
            audio_path: Path to audio file
            graph: BoundaryGraph to add boundaries to
            
        Returns:
            TranscriptData with segments and speech_segments
        """
        self._load_model()
        
        audio_path = Path(audio_path)
        logger.debug(f"Transcribing {audio_path}")
        
        # Transcribe
        segments_iter, info = self._model.transcribe(
            str(audio_path),
            language=self.config.language,
            word_timestamps=True,
            vad_filter=True,
        )
        
        # Process segments
        transcript_segments: List[TranscriptSegment] = []
        speech_segments: List[tuple] = []
        all_words: List[dict] = []
        
        for segment in segments_iter:
            # Segment-level
            transcript_segments.append(TranscriptSegment(
                start=segment.start,
                end=segment.end,
                text=segment.text.strip(),
            ))
            speech_segments.append((segment.start, segment.end))
            
            # Word-level (for sentence boundary detection)
            if segment.words:
                for word in segment.words:
                    all_words.append({
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                    })
        
        logger.info(f"Transcribed {len(transcript_segments)} segments, {len(all_words)} words")
        
        # Detect sentence boundaries from words
        self._detect_sentence_boundaries(all_words, graph)
        
        # Compute speech density at each boundary
        self._annotate_speech_density(graph, all_words)
        
        transcript_data = TranscriptData(
            segments=transcript_segments,
            speech_segments=speech_segments,
        )
        transcript_data.build_index()
        
        return transcript_data
    
    def _detect_sentence_boundaries(
        self,
        words: List[dict],
        graph: BoundaryGraph,
    ) -> List[Boundary]:
        """Detect sentence end boundaries from word timings."""
        boundaries = []
        
        for i, word in enumerate(words):
            text = word["word"].strip()
            
            # Check if word ends with sentence-ending punctuation
            is_sentence_end = any(
                re.search(pattern, text)
                for pattern in self.config.sentence_end_patterns
            )
            
            if not is_sentence_end:
                continue
            
            # Check for pause after this word
            pause_after = 0.0
            if i + 1 < len(words):
                pause_after = words[i + 1]["start"] - word["end"]
            
            # Confidence based on pause duration
            if pause_after >= self.config.min_pause_for_boundary:
                confidence = min(1.0, 0.5 + pause_after)
            else:
                confidence = 0.5
            
            boundary = Boundary(
                timestamp=word["end"],
                types={BoundaryType.SENTENCE_END},
                confidences={BoundaryType.SENTENCE_END: confidence},
                pause_duration=pause_after if pause_after > 0 else None,
            )
            boundaries.append(boundary)
            graph.add(boundary)
        
        logger.debug(f"Found {len(boundaries)} sentence end boundaries")
        
        return boundaries
    
    def _annotate_speech_density(
        self,
        graph: BoundaryGraph,
        words: List[dict],
        window: float = 2.0,
    ):
        """
        Annotate boundaries with speech density (words/sec).
        
        High density = risky cut point (likely mid-speech).
        """
        if not words:
            return
        
        for boundary in graph.boundaries:
            t = boundary.timestamp
            
            # Count words within window
            word_count = sum(
                1 for w in words
                if abs((w["start"] + w["end"]) / 2 - t) < window / 2
            )
            
            density = word_count / window if window > 0 else 0.0
            boundary.speech_density = density
