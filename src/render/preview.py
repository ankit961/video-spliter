"""Low-resolution preview rendering."""

from pathlib import Path
import subprocess
import logging

logger = logging.getLogger(__name__)


# Resolution presets
RESOLUTION_PRESETS = {
    "360p": "640:360",
    "480p": "854:480",
    "720p": "1280:720",
}


def render_preview(
    input_path: Path,
    start_time: float,
    end_time: float,
    output_path: Path,
    resolution: str = "480p",
    crf: int = 28,
    preset: str = "ultrafast",
) -> Path:
    """
    Render a low-resolution preview of a clip.
    
    Args:
        input_path: Source video path
        start_time: Start time in seconds
        end_time: End time in seconds
        output_path: Output video path
        resolution: Resolution preset ("360p", "480p", "720p")
        crf: Quality (higher = smaller file, lower quality)
        preset: Encoding speed preset
        
    Returns:
        Path to rendered preview
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    duration = end_time - start_time
    scale = RESOLUTION_PRESETS.get(resolution, RESOLUTION_PRESETS["480p"])
    
    logger.debug(f"Rendering preview: {start_time:.2f}s - {end_time:.2f}s -> {output_path}")
    
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start_time),
        "-i", str(input_path),
        "-t", str(duration),
        "-vf", f"scale={scale}:force_original_aspect_ratio=decrease,pad={scale}:(ow-iw)/2:(oh-ih)/2",
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", preset,
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        str(output_path),
    ]
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Preview render failed: {result.stderr}")
    
    logger.info(f"Preview rendered: {output_path}")
    
    return output_path
