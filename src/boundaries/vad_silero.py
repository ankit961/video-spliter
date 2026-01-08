"""Voice Activity Detection using Silero VAD."""

from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path
import logging

from ..graph.boundary_graph import Boundary, BoundaryType, BoundaryGraph

logger = logging.getLogger(__name__)


@dataclass
class VADConfig:
    """Configuration for Silero VAD."""
    sample_rate: int = 16000
    min_pause_s: float = 0.35
    pause_boundary_at: str = "start"  # "start" | "mid" | "end"
    min_speech_duration_s: float = 0.1


class SileroVADDetector:
    """
    Voice Activity Detection using Silero VAD.
    
    Produces:
    - Speech segments [(start, end), ...]
    - VAD_PAUSE boundaries at silence gaps
    """
    
    def __init__(self, config: Optional[VADConfig] = None):
        self.config = config or VADConfig()
        self._model = None
        self._utils = None
    
    def _load_model(self):
        """Lazy load Silero VAD model."""
        if self._model is not None:
            return
        
        import torch
        
        self._model, self._utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            trust_repo=True,
        )
        
        (
            self._get_speech_timestamps,
            self._save_audio,
            self._read_audio,
            self._VADIterator,
            self._collect_chunks,
        ) = self._utils
    
    def detect(
        self, 
        audio_path: Path,
        graph: BoundaryGraph,
    ) -> Tuple[List[Tuple[float, float]], List[Boundary]]:
        """
        Detect speech segments and pause boundaries.
        
        Args:
            audio_path: Path to audio file (wav, mp3, etc.)
            graph: BoundaryGraph to add boundaries to
            
        Returns:
            (speech_segments, boundaries)
        """
        self._load_model()
        
        audio_path = Path(audio_path)
        logger.debug(f"Running VAD on {audio_path}")
        
        # Read audio
        wav = self._read_audio(str(audio_path), sampling_rate=self.config.sample_rate)
        
        # Get speech timestamps
        speech_ts = self._get_speech_timestamps(
            wav,
            self._model,
            sampling_rate=self.config.sample_rate,
            min_speech_duration_ms=int(self.config.min_speech_duration_s * 1000),
        )
        
        # Convert to seconds
        speech_segments: List[Tuple[float, float]] = []
        for seg in speech_ts:
            start = seg["start"] / self.config.sample_rate
            end = seg["end"] / self.config.sample_rate
            if end > start:
                speech_segments.append((start, end))
        
        # Derive pause boundaries
        boundaries: List[Boundary] = []
        prev_end = 0.0
        
        for start, end in speech_segments:
            pause_duration = start - prev_end
            
            if pause_duration >= self.config.min_pause_s:
                # Determine boundary timestamp
                if self.config.pause_boundary_at == "mid":
                    t = prev_end + pause_duration / 2
                elif self.config.pause_boundary_at == "end":
                    t = start
                else:  # "start"
                    t = prev_end
                
                boundary = Boundary(
                    timestamp=float(t),
                    types={BoundaryType.VAD_PAUSE},
                    confidences={BoundaryType.VAD_PAUSE: 1.0},
                    pause_duration=float(pause_duration),
                )
                boundaries.append(boundary)
                graph.add(boundary)
            
            prev_end = end
        
        # Handle trailing silence
        if graph.video_duration > 0 and prev_end < graph.video_duration:
            trailing_silence = graph.video_duration - prev_end
            if trailing_silence >= self.config.min_pause_s:
                boundary = Boundary(
                    timestamp=float(prev_end),
                    types={BoundaryType.VAD_PAUSE},
                    confidences={BoundaryType.VAD_PAUSE: 1.0},
                    pause_duration=float(trailing_silence),
                )
                boundaries.append(boundary)
                graph.add(boundary)
        
        logger.info(f"VAD: {len(speech_segments)} speech segments, {len(boundaries)} pause boundaries")
        
        return speech_segments, boundaries
