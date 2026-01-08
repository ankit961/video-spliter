"""Shot boundary detection using TransNetV2 or PySceneDetect."""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path
import logging

import numpy as np

from ..graph.boundary_graph import Boundary, BoundaryType, BoundaryGraph

logger = logging.getLogger(__name__)


@dataclass
class TransNetConfig:
    """Configuration for shot detection."""
    threshold: float = 0.5
    min_shot_length_s: float = 0.5
    use_pyscenedetect: bool = True  # Fallback if TransNetV2 not available
    pyscenedetect_threshold: float = 27.0


class TransNetV2ShotDetector:
    """
    Shot boundary detection.
    
    Uses PySceneDetect as primary (more reliable), with TransNetV2 as option.
    """
    
    def __init__(self, config: Optional[TransNetConfig] = None):
        self.config = config or TransNetConfig()
        self._model = None
        self._use_transnet = False
    
    def _load_model(self):
        """Try to load TransNetV2, fall back to PySceneDetect."""
        if self._model is not None:
            return
        
        if not self.config.use_pyscenedetect:
            try:
                self._load_transnet()
                self._use_transnet = True
                return
            except Exception as e:
                logger.warning(f"TransNetV2 not available: {e}, using PySceneDetect")
        
        # PySceneDetect is always available
        self._use_transnet = False
        logger.info("Using PySceneDetect for shot detection")
    
    def _load_transnet(self):
        """Load TransNetV2 model."""
        try:
            import torch
            # Try to import TransNetV2
            # Note: Requires separate installation
            from transnetv2 import TransNetV2
            self._model = TransNetV2()
            logger.info("Loaded TransNetV2 model")
        except ImportError:
            raise ImportError("TransNetV2 not installed. Use PySceneDetect instead.")
    
    def detect(
        self,
        video_path: Path,
        graph: BoundaryGraph,
    ) -> List[Boundary]:
        """
        Detect shot boundaries in video.
        
        Args:
            video_path: Path to video file
            graph: BoundaryGraph to add boundaries to
            
        Returns:
            List of shot boundaries
        """
        self._load_model()
        
        video_path = Path(video_path)
        logger.debug(f"Detecting shots in {video_path}")
        
        if self._use_transnet:
            return self._detect_transnet(video_path, graph)
        else:
            return self._detect_pyscenedetect(video_path, graph)
    
    def _detect_pyscenedetect(
        self,
        video_path: Path,
        graph: BoundaryGraph,
    ) -> List[Boundary]:
        """Detect shots using PySceneDetect."""
        from scenedetect import detect, ContentDetector
        
        # Detect scenes
        scene_list = detect(
            str(video_path),
            ContentDetector(threshold=self.config.pyscenedetect_threshold),
        )
        
        boundaries = []
        prev_time = 0.0
        
        for scene in scene_list:
            start_time = scene[0].get_seconds()
            
            # Check minimum shot length
            if start_time - prev_time < self.config.min_shot_length_s:
                continue
            
            # Skip the very first frame
            if start_time < 0.1:
                continue
            
            boundary = Boundary(
                timestamp=start_time,
                types={BoundaryType.SHOT},
                confidences={BoundaryType.SHOT: 0.9},  # PySceneDetect doesn't give confidence
            )
            boundaries.append(boundary)
            graph.add(boundary)
            
            prev_time = start_time
        
        logger.info(f"Found {len(boundaries)} shot boundaries (PySceneDetect)")
        
        return boundaries
    
    def _detect_transnet(
        self,
        video_path: Path,
        graph: BoundaryGraph,
    ) -> List[Boundary]:
        """Detect shots using TransNetV2."""
        # Get video frames and FPS
        frames, fps = self._extract_frames(video_path)
        
        if len(frames) == 0:
            return []
        
        # Run TransNetV2
        predictions = self._model.predict_frames(frames)
        
        # predictions shape: (n_frames, 2) — softmax over [no_cut, cut]
        cut_probs = predictions[:, 1]
        
        # Find shot boundaries above threshold
        boundaries = []
        prev_cut_frame = 0
        
        for frame_idx, prob in enumerate(cut_probs):
            if prob < self.config.threshold:
                continue
            
            # Check minimum shot length
            time_since_last = (frame_idx - prev_cut_frame) / fps
            if time_since_last < self.config.min_shot_length_s:
                continue
            
            timestamp = frame_idx / fps
            
            boundary = Boundary(
                timestamp=timestamp,
                types={BoundaryType.SHOT},
                confidences={BoundaryType.SHOT: float(prob)},
            )
            boundaries.append(boundary)
            graph.add(boundary)
            
            prev_cut_frame = frame_idx
        
        logger.info(f"Found {len(boundaries)} shot boundaries (TransNetV2)")
        
        return boundaries
    
    def _extract_frames(
        self,
        video_path: Path,
    ) -> Tuple[np.ndarray, float]:
        """Extract frames from video for TransNetV2."""
        import cv2
        
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # TransNetV2 expects 48x27 RGB frames
            frame = cv2.resize(frame, (48, 27))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        if not frames:
            return np.array([]), fps
        
        return np.array(frames), fps
