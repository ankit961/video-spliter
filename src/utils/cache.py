"""Pipeline caching utilities."""

from dataclasses import dataclass
from typing import Any, Callable, Optional
from pathlib import Path
import pickle
import hashlib
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for pipeline cache."""
    enabled: bool = True
    cache_dir: Path = Path(".cache")


class PipelineCache:
    """
    Simple file-based cache for pipeline artifacts.
    
    Caches:
    - Boundary graphs
    - Transcripts
    - Speech segments
    """
    
    def __init__(self, cache_dir: Optional[Path] = None, enabled: bool = True):
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".cache")
        self.enabled = enabled
        
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], Any],
    ) -> Any:
        """
        Get cached value or compute and cache it.
        
        Args:
            key: Cache key (will be hashed)
            compute_fn: Function to compute value if not cached
            
        Returns:
            Cached or computed value
        """
        if not self.enabled:
            return compute_fn()
        
        cache_path = self._get_cache_path(key)
        
        # Try to load from cache
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    value = pickle.load(f)
                logger.debug(f"Cache hit: {key}")
                return value
            except Exception as e:
                logger.warning(f"Cache load failed for {key}: {e}")
        
        # Compute and cache
        logger.debug(f"Cache miss: {key}")
        value = compute_fn()
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
            logger.debug(f"Cached: {key}")
        except Exception as e:
            logger.warning(f"Cache save failed for {key}: {e}")
        
        return value
    
    def invalidate(self, key: str):
        """Invalidate a cache entry."""
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            cache_path.unlink()
            logger.debug(f"Cache invalidated: {key}")
    
    def clear(self):
        """Clear all cache entries."""
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            logger.info("Cache cleared")
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a key."""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.pkl"


def compute_video_hash(video_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """
    Compute hash of video file for cache keying.
    
    Only reads first chunk_size bytes for speed.
    """
    with open(video_path, 'rb') as f:
        data = f.read(chunk_size)
    return hashlib.md5(data).hexdigest()[:12]
