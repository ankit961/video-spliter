"""Full-quality final rendering with fade support."""

from pathlib import Path
import subprocess
import logging

logger = logging.getLogger(__name__)


def render_final(
    input_path: Path,
    start_time: float,
    end_time: float,
    output_path: Path,
    codec: str = "libx264",
    crf: int = 18,
    preset: str = "medium",
    audio_bitrate: str = "192k",
    copy_streams: bool = False,
    fade_in_ms: int = 0,
    fade_out_ms: int = 0,
) -> Path:
    """
    Render a full-quality final clip with optional fades.
    
    Args:
        input_path: Source video path
        start_time: Start time in seconds
        end_time: End time in seconds
        output_path: Output video path
        codec: Video codec (libx264, libx265, etc.)
        crf: Quality (lower = higher quality)
        preset: Encoding speed preset
        audio_bitrate: Audio bitrate
        copy_streams: If True, copy streams without re-encoding
        fade_in_ms: Audio fade-in duration in milliseconds
        fade_out_ms: Audio fade-out duration in milliseconds
        
    Returns:
        Path to rendered clip
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    duration = end_time - start_time
    
    logger.debug(f"Rendering final: {start_time:.2f}s - {end_time:.2f}s -> {output_path}")
    
    # Build audio filter for fades
    audio_filters = []
    if fade_in_ms > 0:
        fade_in_s = fade_in_ms / 1000.0
        audio_filters.append(f"afade=t=in:st=0:d={fade_in_s}")
    if fade_out_ms > 0:
        fade_out_s = fade_out_ms / 1000.0
        fade_start = duration - fade_out_s
        audio_filters.append(f"afade=t=out:st={fade_start}:d={fade_out_s}")
    
    # Build video filter for fades
    video_filters = []
    if fade_in_ms > 0:
        fade_in_s = fade_in_ms / 1000.0
        video_filters.append(f"fade=t=in:st=0:d={fade_in_s}")
    if fade_out_ms > 0:
        fade_out_s = fade_out_ms / 1000.0
        fade_start = duration - fade_out_s
        video_filters.append(f"fade=t=out:st={fade_start}:d={fade_out_s}")
    
    if copy_streams and not audio_filters and not video_filters:
        # Fast copy mode (may have keyframe alignment issues)
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start_time),
            "-i", str(input_path),
            "-t", str(duration),
            "-c", "copy",
            "-movflags", "+faststart",
            str(output_path),
        ]
    else:
        # Re-encode for precise cuts and fades
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start_time),
            "-i", str(input_path),
            "-t", str(duration),
        ]
        
        # Add video filter
        if video_filters:
            cmd.extend(["-vf", ",".join(video_filters)])
        
        cmd.extend([
            "-c:v", codec,
            "-crf", str(crf),
            "-preset", preset,
        ])
        
        # Add audio filter
        if audio_filters:
            cmd.extend(["-af", ",".join(audio_filters)])
        
        cmd.extend([
            "-c:a", "aac",
            "-b:a", audio_bitrate,
            "-movflags", "+faststart",
            str(output_path),
        ])
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Final render failed: {result.stderr}")
    
    logger.info(f"Final rendered: {output_path}")
    
    return output_path


def render_with_captions(
    input_path: Path,
    start_time: float,
    end_time: float,
    output_path: Path,
    srt_path: Path,
    codec: str = "libx264",
    crf: int = 18,
    preset: str = "medium",
    font_size: int = 24,
    font_color: str = "white",
    outline_color: str = "black",
) -> Path:
    """
    Render a clip with burned-in captions.
    
    Args:
        input_path: Source video path
        start_time: Start time in seconds
        end_time: End time in seconds
        output_path: Output video path
        srt_path: Path to SRT subtitle file
        codec: Video codec
        crf: Quality
        preset: Encoding preset
        font_size: Caption font size
        font_color: Caption font color
        outline_color: Caption outline color
        
    Returns:
        Path to rendered clip
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    srt_path = Path(srt_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    duration = end_time - start_time
    
    # Escape path for ffmpeg filter
    srt_escaped = str(srt_path).replace(":", r"\:").replace("'", r"\'")
    
    subtitle_filter = (
        f"subtitles='{srt_escaped}':force_style='"
        f"FontSize={font_size},"
        f"PrimaryColour=&H{_color_to_ass(font_color)},"
        f"OutlineColour=&H{_color_to_ass(outline_color)},"
        f"BorderStyle=1,Outline=2'"
    )
    
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start_time),
        "-i", str(input_path),
        "-t", str(duration),
        "-vf", subtitle_filter,
        "-c:v", codec,
        "-crf", str(crf),
        "-preset", preset,
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        str(output_path),
    ]
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Captioned render failed: {result.stderr}")
    
    logger.info(f"Captioned clip rendered: {output_path}")
    
    return output_path


def _color_to_ass(color: str) -> str:
    """Convert color name to ASS format (BGR with alpha)."""
    colors = {
        "white": "00FFFFFF",
        "black": "00000000",
        "yellow": "0000FFFF",
        "red": "000000FF",
        "blue": "00FF0000",
        "green": "0000FF00",
    }
    return colors.get(color.lower(), "00FFFFFF")
