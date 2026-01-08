"""Speaker turn detection using pyannote.audio."""

from dataclasses import dataclass
from typing import List, Optional, Dict
from pathlib import Path
import logging

from ..graph.boundary_graph import Boundary, BoundaryType, BoundaryGraph

logger = logging.getLogger(__name__)


@dataclass
class PyannoteConfig:
    """Configuration for pyannote speaker diarization."""
    hf_token: Optional[str] = None  # HuggingFace token for model access
    min_speakers: int = 1
    max_speakers: Optional[int] = None
    min_turn_duration: float = 1.0


class PyannoteSpeakerDetector:
    """
    Speaker turn detection using pyannote.audio.
    
    Detects when speakers change, useful for interview/podcast content.
    """
    
    def __init__(self, config: Optional[PyannoteConfig] = None):
        self.config = config or PyannoteConfig()
        self._pipeline = None
    
    def _load_model(self):
        """Lazy load pyannote pipeline."""
        if self._pipeline is not None:
            return
        
        try:
            from pyannote.audio import Pipeline
            import torch
            
            logger.info("Loading pyannote speaker diarization model")
            
            # Load pretrained pipeline
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.config.hf_token,
            )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self._pipeline.to(torch.device("cuda"))
                
        except ImportError:
            raise ImportError(
                "pyannote.audio not installed. Install with: pip install pyannote.audio"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load pyannote model. Ensure you have accepted the model terms "
                f"on HuggingFace and provided a valid token. Error: {e}"
            )
    
    def detect(
        self,
        audio_path: Path,
        graph: BoundaryGraph,
    ) -> List[Boundary]:
        """
        Detect speaker turn boundaries.
        
        Args:
            audio_path: Path to audio file
            graph: BoundaryGraph to add boundaries to
            
        Returns:
            List of speaker turn boundaries
        """
        self._load_model()
        
        audio_path = Path(audio_path)
        logger.debug(f"Running speaker diarization on {audio_path}")
        
        # Run diarization
        diarization = self._pipeline(
            str(audio_path),
            min_speakers=self.config.min_speakers,
            max_speakers=self.config.max_speakers,
        )
        
        # Extract speaker turns
        turns: List[tuple] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if turn.duration >= self.config.min_turn_duration:
                turns.append((turn.start, turn.end, speaker))
        
        # Create boundaries at speaker changes
        boundaries = []
        prev_speaker = None
        
        for start, end, speaker in turns:
            if prev_speaker is not None and speaker != prev_speaker:
                boundary = Boundary(
                    timestamp=start,
                    types={BoundaryType.SPEAKER_TURN},
                    confidences={BoundaryType.SPEAKER_TURN: 0.9},
                )
                boundaries.append(boundary)
                graph.add(boundary)
            
            prev_speaker = speaker
        
        logger.info(f"Found {len(boundaries)} speaker turn boundaries")
        
        return boundaries
    
    def get_speaker_segments(
        self,
        audio_path: Path,
    ) -> Dict[str, List[tuple]]:
        """
        Get all segments per speaker.
        
        Returns:
            {"SPEAKER_00": [(start, end), ...], ...}
        """
        self._load_model()
        
        diarization = self._pipeline(
            str(audio_path),
            min_speakers=self.config.min_speakers,
            max_speakers=self.config.max_speakers,
        )
        
        speaker_segments: Dict[str, List[tuple]] = {}
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append((turn.start, turn.end))
        
        return speaker_segments
