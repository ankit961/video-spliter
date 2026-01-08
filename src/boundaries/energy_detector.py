"""Energy-based boundary detection for finding quiet moments."""

from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path
import logging
import numpy as np

from ..graph.boundary_graph import Boundary, BoundaryType, BoundaryGraph

logger = logging.getLogger(__name__)


@dataclass
class EnergyConfig:
    """Configuration for energy-based detection."""
    sample_rate: int = 16000
    frame_size_ms: int = 25
    hop_size_ms: int = 10
    
    # Energy dip detection
    min_dip_duration_ms: int = 100
    dip_threshold_percentile: float = 25.0  # Below this percentile = quiet
    
    # Smoothing
    smoothing_window_ms: int = 200
    
    # Output
    max_boundaries_per_minute: float = 10.0


class EnergyBoundaryDetector:
    """
    Detects low-energy moments in audio for clean cut points.
    
    Especially useful for music content where VAD might not 
    detect pauses but there are still quieter moments.
    """
    
    def __init__(self, config: Optional[EnergyConfig] = None):
        self.config = config or EnergyConfig()
    
    def detect(
        self,
        audio_path: Path,
        graph: BoundaryGraph,
    ) -> List[Boundary]:
        """
        Detect energy dip boundaries.
        
        Args:
            audio_path: Path to audio file
            graph: BoundaryGraph (for duration info)
            
        Returns:
            List of ENERGY_DIP boundaries
        """
        import torchaudio
        
        audio_path = Path(audio_path)
        logger.debug(f"Running energy detection on {audio_path}")
        
        # Load audio
        waveform, sr = torchaudio.load(str(audio_path))
        
        # Resample if needed
        if sr != self.config.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.config.sample_rate)
            waveform = resampler(waveform)
            sr = self.config.sample_rate
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        audio = waveform.squeeze().numpy()
        
        # Compute frame-level energy
        frame_size = int(self.config.frame_size_ms * sr / 1000)
        hop_size = int(self.config.hop_size_ms * sr / 1000)
        
        energy = self._compute_energy(audio, frame_size, hop_size)
        
        # Smooth energy curve
        smooth_window = int(self.config.smoothing_window_ms / self.config.hop_size_ms)
        if smooth_window > 1:
            energy = self._smooth(energy, smooth_window)
        
        # Find energy dips (local minima below threshold)
        threshold = np.percentile(energy, self.config.dip_threshold_percentile)
        dip_indices = self._find_dips(energy, threshold)
        
        # Convert to time
        boundaries: List[Boundary] = []
        min_dip_frames = int(self.config.min_dip_duration_ms / self.config.hop_size_ms)
        
        for idx in dip_indices:
            time_s = idx * self.config.hop_size_ms / 1000.0
            
            # Skip if too close to start/end
            if time_s < 1.0 or time_s > graph.video_duration - 1.0:
                continue
            
            # Compute confidence based on how low the energy is
            energy_val = energy[idx]
            confidence = 1.0 - (energy_val / (threshold + 1e-8))
            confidence = max(0.3, min(1.0, confidence))
            
            boundary = Boundary(
                timestamp=time_s,
                types={BoundaryType.ENERGY_DIP},
                confidences={BoundaryType.ENERGY_DIP: confidence},
                energy_slope=self._compute_slope(energy, idx, 10),
            )
            boundaries.append(boundary)
        
        # Limit density
        max_boundaries = int(graph.video_duration / 60.0 * self.config.max_boundaries_per_minute)
        if len(boundaries) > max_boundaries:
            # Keep highest confidence ones
            boundaries.sort(key=lambda b: b.confidence, reverse=True)
            boundaries = boundaries[:max_boundaries]
            boundaries.sort(key=lambda b: b.timestamp)
        
        logger.info(f"Energy: {len(boundaries)} dip boundaries")
        return boundaries
    
    def _compute_energy(self, audio: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
        """Compute RMS energy per frame."""
        n_frames = (len(audio) - frame_size) // hop_size + 1
        energy = np.zeros(n_frames)
        
        for i in range(n_frames):
            start = i * hop_size
            frame = audio[start:start + frame_size]
            energy[i] = np.sqrt(np.mean(frame ** 2))
        
        return energy
    
    def _smooth(self, signal: np.ndarray, window: int) -> np.ndarray:
        """Apply moving average smoothing."""
        kernel = np.ones(window) / window
        return np.convolve(signal, kernel, mode='same')
    
    def _find_dips(self, energy: np.ndarray, threshold: float) -> List[int]:
        """Find local minima below threshold."""
        dips = []
        
        # Find all points below threshold
        below = energy < threshold
        
        # Find local minima
        for i in range(1, len(energy) - 1):
            if below[i] and energy[i] <= energy[i-1] and energy[i] <= energy[i+1]:
                # Check it's a "real" dip (not just noise)
                if i >= 5 and i < len(energy) - 5:
                    local_max = max(energy[i-5:i+5])
                    if energy[i] < local_max * 0.7:  # Significant dip
                        dips.append(i)
        
        # Merge nearby dips (keep the lowest)
        if not dips:
            return dips
        
        merged = [dips[0]]
        min_gap_frames = 50  # ~500ms
        
        for idx in dips[1:]:
            if idx - merged[-1] < min_gap_frames:
                # Keep the one with lower energy
                if energy[idx] < energy[merged[-1]]:
                    merged[-1] = idx
            else:
                merged.append(idx)
        
        return merged
    
    def _compute_slope(self, energy: np.ndarray, idx: int, window: int) -> float:
        """Compute energy slope at a point (negative = decreasing)."""
        start = max(0, idx - window)
        end = min(len(energy), idx + window)
        
        if end - start < 3:
            return 0.0
        
        x = np.arange(end - start)
        y = energy[start:end]
        
        # Simple linear regression slope
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)
