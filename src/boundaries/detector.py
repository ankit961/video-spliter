"""Unified boundary detection combining multiple signals."""

from dataclasses import dataclass
from typing import Optional, List, Tuple
from pathlib import Path
import subprocess
import tempfile
import logging

from ..graph.boundary_graph import BoundaryGraph
from ..utils.transcript import TranscriptData

from .vad_silero import SileroVADDetector, VADConfig
from .transcript_whisper import WhisperTranscriptDetector, WhisperConfig
from .shot_transnet import TransNetV2ShotDetector, TransNetConfig
from .speaker_pyannote import PyannoteSpeakerDetector, PyannoteConfig

logger = logging.getLogger(__name__)


@dataclass
class DetectorConfig:
    """Configuration for unified boundary detection."""
    # Which detectors to run
    use_vad: bool = True
    use_transcript: bool = True
    use_shot: bool = True
    use_speaker: bool = False  # Optional, slower
    
    # Detector configs
    vad_config: Optional[VADConfig] = None
    whisper_config: Optional[WhisperConfig] = None
    transnet_config: Optional[TransNetConfig] = None
    pyannote_config: Optional[PyannoteConfig] = None


class UnifiedBoundaryDetector:
    """
    Unified boundary detection combining multiple signals.
    
    Runs configured detectors and populates a BoundaryGraph.
    """
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
        
        # Initialize detectors lazily
        self._vad_detector = None
        self._transcript_detector = None
        self._shot_detector = None
        self._speaker_detector = None
        
        # Cached results
        self._last_transcript: Optional[TranscriptData] = None
        self._last_speech_segments: Optional[List[Tuple[float, float]]] = None
    
    @property
    def vad_detector(self) -> SileroVADDetector:
        if self._vad_detector is None:
            self._vad_detector = SileroVADDetector(self.config.vad_config)
        return self._vad_detector
    
    @property
    def transcript_detector(self) -> WhisperTranscriptDetector:
        if self._transcript_detector is None:
            self._transcript_detector = WhisperTranscriptDetector(self.config.whisper_config)
        return self._transcript_detector
    
    @property
    def shot_detector(self) -> TransNetV2ShotDetector:
        if self._shot_detector is None:
            self._shot_detector = TransNetV2ShotDetector(self.config.transnet_config)
        return self._shot_detector
    
    @property
    def speaker_detector(self) -> PyannoteSpeakerDetector:
        if self._speaker_detector is None:
            self._speaker_detector = PyannoteSpeakerDetector(self.config.pyannote_config)
        return self._speaker_detector
    
    def detect_all(self, video_path: Path) -> BoundaryGraph:
        """
        Run all configured detectors and return populated BoundaryGraph.
        
        Note: Does NOT call finalize() — caller should do that after
        optionally adding stitch marks.
        """
        video_path = Path(video_path)
        
        # Get video duration
        duration = self._get_video_duration(video_path)
        logger.info(f"Video duration: {duration:.2f}s")
        
        graph = BoundaryGraph(video_duration=duration)
        
        # Extract audio for audio-based detectors
        audio_path = None
        
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                audio_path = Path(f.name)
            
            self._extract_audio(video_path, audio_path)
            
            # Run VAD first (provides speech segments for other detectors)
            if self.config.use_vad:
                logger.info("Running VAD detection...")
                speech_segments, vad_boundaries = self.vad_detector.detect(
                    audio_path, graph
                )
                self._last_speech_segments = speech_segments
            
            # Run transcript detection
            if self.config.use_transcript:
                logger.info("Running transcript detection...")
                transcript_data = self.transcript_detector.transcribe(
                    audio_path, graph
                )
                self._last_transcript = transcript_data
                
                # Update speech segments if VAD wasn't run
                if self._last_speech_segments is None:
                    self._last_speech_segments = transcript_data.speech_segments
            
            # Run speaker detection
            if self.config.use_speaker:
                logger.info("Running speaker detection...")
                try:
                    self.speaker_detector.detect(audio_path, graph)
                except Exception as e:
                    logger.warning(f"Speaker detection failed: {e}")
        
        finally:
            # Clean up temp audio
            if audio_path and audio_path.exists():
                audio_path.unlink()
        
        # Run shot detection (needs video, not audio)
        if self.config.use_shot:
            logger.info("Running shot detection...")
            try:
                self.shot_detector.detect(video_path, graph)
            except Exception as e:
                logger.warning(f"Shot detection failed: {e}")
        
        logger.info(f"Total boundaries detected: {len(graph.boundaries)}")
        
        return graph
    
    def transcribe(self, video_path: Path) -> TranscriptData:
        """
        Get transcript data (runs detection if not cached).
        """
        if self._last_transcript is not None:
            return self._last_transcript
        
        video_path = Path(video_path)
        audio_path = None
        
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                audio_path = Path(f.name)
            
            self._extract_audio(video_path, audio_path)
            
            # Create temporary graph for transcript detection
            duration = self._get_video_duration(video_path)
            temp_graph = BoundaryGraph(video_duration=duration)
            
            transcript_data = self.transcript_detector.transcribe(
                audio_path, temp_graph
            )
            self._last_transcript = transcript_data
            
            # Also get speech segments from VAD if available
            if self.config.use_vad and self._last_speech_segments is None:
                speech_segments, _ = self.vad_detector.detect(audio_path, temp_graph)
                self._last_speech_segments = speech_segments
                transcript_data.speech_segments = speech_segments
            
            return transcript_data
        
        finally:
            if audio_path and audio_path.exists():
                audio_path.unlink()
    
    def get_speech_segments(self) -> List[Tuple[float, float]]:
        """Get cached speech segments."""
        return self._last_speech_segments or []
    
    def _get_video_duration(self, video_path: Path) -> float:
        """Get video duration using ffprobe."""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr}")
        
        return float(result.stdout.strip())
    
    def _extract_audio(
        self, 
        video_path: Path, 
        audio_path: Path,
        sample_rate: int = 16000,
    ):
        """Extract audio from video using ffmpeg."""
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(sample_rate),
            "-ac", "1",
            str(audio_path),
        ]
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr}")
