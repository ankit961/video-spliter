# Video Split Pipeline v2

Production-grade video segmentation using a **boundary graph + dynamic programming** approach instead of naive LLM-driven editing.

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/ankit961/video-spliter.git
cd video-spliter

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install ffmpeg (required)
# macOS: brew install ffmpeg
# Ubuntu: sudo apt install ffmpeg

# 5. Run the pipeline
python main.py your_video.mp4 --mode coverage
```

## ✨ Key Features

- **Deterministic boundaries** — Shot detection + VAD + sentence endings for reliable cut points
- **Smooth transitions** — Audio/video fades (300ms in, 400ms out) for professional results
- **Multi-signal alignment** — Prefers cuts where shot, sentence, and speech pause align
- **Full coverage** — DP optimization ensures 95%+ video coverage with minimal gaps
- **Smart speech detection** — Automatically prefers transcript-based segments for singing/music content
- **Energy-based boundaries** — Detects audio energy dips for natural cut points

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        VIDEO INPUT                               │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BOUNDARY DETECTION                            │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌───────┐ │
│  │ Silero  │  │ Whisper │  │ PyScene │  │Pyannote │  │Energy │ │
│  │   VAD   │  │ Transcr.│  │ Detect  │  │ Speaker │  │  Dip  │ │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └───┬───┘ │
│       │            │            │            │           │      │
│       └────────────┴────────────┴────────────┴───────────┘      │
│                          │                                       │
│                          ▼                                       │
│               ┌─────────────────────┐                            │
│               │   BOUNDARY GRAPH    │                            │
│               │  (merge + finalize) │                            │
│               └─────────────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CANDIDATE GENERATION                          │
│         Generate all valid 45-60s clips from graph               │
│              (windowed O(n·k) edge generation)                   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       OPTIMIZATION                               │
│  ┌─────────────────────┐    ┌─────────────────────┐             │
│  │   DP Coverage       │ OR │  Greedy Highlights  │             │
│  │ (backpointer-based) │    │   (top-k by score)  │             │
│  └─────────────────────┘    └─────────────────────┘             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      REVIEW LOOP                                 │
│        ┌──────────────┐         ┌──────────────┐                │
│        │  Heuristic   │──PASS──▶│   APPROVED   │                │
│        │   Checks     │         └──────────────┘                │
│        └──────┬───────┘                                         │
│               │BORDERLINE                                        │
│               ▼                                                  │
│        ┌──────────────┐         ┌──────────────┐                │
│        │  LLM Review  │──PASS──▶│   APPROVED   │                │
│        │ (structured) │         └──────────────┘                │
│        └──────┬───────┘                                         │
│               │FAIL + SHIFT                                      │
│               ▼                                                  │
│        ┌──────────────┐                                         │
│        │ Apply Shifts │──────▶ (retry with adjusted clip)       │
│        └──────────────┘                                         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        RENDER                                    │
│      Preview (low-res) + Final (full quality with fades)         │
└─────────────────────────────────────────────────────────────────┘
```

## Key Design Principles

1. **Deterministic boundaries** — Shot detection + VAD gives structural truth
2. **Boundary graph abstraction** — Decouples "where can I cut" from "where should I cut"
3. **DP selection over greedy** — Globally optimal clip selection
4. **Heuristics-first filtering** — 80%+ decisions don't need an LLM
5. **Structured outputs** — LLM can only PASS/FAIL and shift to adjacent boundaries
6. **Smooth transitions** — Audio/video fades prevent jarring cuts

## Installation

```bash
# Clone repository
git clone https://github.com/ankit961/video-spliter.git
cd video-spliter

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Install ffmpeg (required)
# macOS: brew install ffmpeg
# Ubuntu: sudo apt install ffmpeg
# Windows: Download from https://ffmpeg.org/download.html
```

### Optional: GPU Acceleration

```bash
# For CUDA support
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Optional: Speaker Diarization

Speaker diarization requires accepting the pyannote model terms:
1. Visit https://huggingface.co/pyannote/speaker-diarization-3.1
2. Accept the terms
3. Set your HuggingFace token: `export HF_TOKEN=your_token`

## Usage

### Basic Usage

```bash
# Full coverage mode (default) - splits entire video into clips
python main.py video.mp4

# Highlights mode - extracts best clips only
python main.py video.mp4 --mode highlights --max-clips 5
```

### Custom Duration Range

```bash
# 30-45 second clips
python main.py video.mp4 --min-duration 30 --max-duration 45
```

### With Stitch Marks

```bash
# Stitch marks as forbidden zones (default)
python main.py video.mp4 --stitch-marks 60.5,120.0,180.5

# Stitch marks as preferred cut points
python main.py video.mp4 --stitch-marks 60.5,120.0 --stitch-behavior preferred

# Stitch marks as mandatory cut points
python main.py video.mp4 --stitch-marks 60.5,120.0 --stitch-behavior must-cut
```

### Disable Features

```bash
# No LLM review (faster, heuristics only)
python main.py video.mp4 --no-llm

# No shot detection
python main.py video.mp4 --no-shot

# Minimal detection (just VAD)
python main.py video.mp4 --no-transcript --no-shot
```

### Output Options

```bash
# Custom output directory
python main.py video.mp4 --output clips/

# Verbose logging
python main.py video.mp4 -v
```

## Project Structure

```
video-spliter/
├── main.py                      # CLI entry point
├── requirements.txt             # Dependencies
├── config/
│   └── pipeline_config.yaml     # Configuration template
├── src/
│   ├── pipeline.py              # Main orchestrator
│   ├── boundaries/
│   │   ├── detector.py          # Unified boundary detection
│   │   ├── vad_silero.py        # Voice activity detection
│   │   ├── transcript_whisper.py # Transcription + sentence ends
│   │   ├── shot_transnet.py     # Shot boundary detection
│   │   ├── energy_detector.py   # Audio energy dip detection
│   │   └── speaker_pyannote.py  # Speaker turn detection
│   ├── graph/
│   │   ├── boundary_graph.py    # Graph data structure
│   │   ├── candidate_edges.py   # Clip candidate generation
│   │   └── scorer.py            # Clip scoring with multi-signal alignment
│   ├── optimizer/
│   │   ├── dp_coverage.py       # DP for full coverage
│   │   └── greedy_highlights.py # Top-k selection
│   ├── reviewer/
│   │   ├── heuristic_checks.py  # Fast PASS/FAIL rules
│   │   └── llm_reviewer.py      # Structured LLM review
│   ├── render/
│   │   ├── preview.py           # Low-res preview
│   │   └── final.py             # Full quality export with fades
│   └── utils/
│       ├── transcript.py        # Transcript data structure
│       └── cache.py             # Pipeline caching
└── output/                      # Default output directory
```

## Configuration

Edit `config/pipeline_config.yaml` to customize behavior. Key settings:

### Duration

```yaml
pipeline:
  min_clip_duration: 45.0
  max_clip_duration: 60.0
```

### Coverage vs Highlights

```yaml
pipeline:
  mode: coverage  # or 'highlights'
  max_clips: 10   # for highlights mode
```

### Smooth Transitions

```yaml
render:
  fade_in_ms: 300   # Audio/video fade in duration
  fade_out_ms: 400  # Audio/video fade out duration
```

### Scoring Weights

```yaml
scoring:
  boundary_weights:
    shot: 1.0         # Visual cuts (highest)
    vad_pause: 0.9    # Speech pauses
    sentence_end: 0.7 # Sentence boundaries
    energy_dip: 0.5   # Audio energy dips
  
  # Multi-signal alignment bonuses
  sentence_alignment_bonus: 1.5  # Bonus for shot + sentence alignment
  audio_abrupt_penalty: 0.5      # Penalty for cuts during speech
```

### LLM Settings

```yaml
review:
  use_llm: true
  llm_model: gpt-4o-mini
```

## Python API

```python
from pathlib import Path
from src.pipeline import VideoSplitPipeline, PipelineConfig
from src.graph.boundary_graph import StitchBehavior

# Create config
config = PipelineConfig(
    mode="coverage",
    min_clip_duration=45.0,
    max_clip_duration=60.0,
    use_llm_review=True,
    output_dir=Path("output"),
    fade_in_ms=300,     # Smooth fade in
    fade_out_ms=400,    # Smooth fade out
)

# Create pipeline
pipeline = VideoSplitPipeline(config)

# Process video
results = pipeline.process(
    Path("video.mp4"),
    stitch_marks=[60.0, 120.0, 180.0],
)

# Access results
for result in results:
    if result.passed_review:
        print(f"Clip: {result.clip.start:.1f}s - {result.clip.end:.1f}s")
        print(f"  Path: {result.final_path}")
        print(f"  Transcript: {result.transcript[:100]}...")
```

## How It Works

### 1. Boundary Detection

Multiple detectors run in parallel to find potential cut points:

- **Silero VAD**: Detects speech pauses (most reliable for talking-head content)
- **Whisper**: Transcribes audio and detects sentence endings
- **PySceneDetect**: Detects visual shot boundaries
- **Energy Detector**: Finds audio energy dips for natural transitions
- **Pyannote**: Detects speaker changes (optional)

### 2. Boundary Graph

All detected boundaries are merged into a unified graph:

- Nearby boundaries (< 500ms) are merged
- Anchor boundaries (VIDEO_START, VIDEO_END, STITCH_MARK) preserve exact timestamps
- Multi-signal boundaries (detected by multiple methods) get bonus scores

### 3. Candidate Generation

Valid clip candidates are generated using windowed iteration:

- Only clips within duration range (e.g., 45-60s) are considered
- O(n·k) complexity where k = boundaries per 15s window
- MUST_CUT mode enforces clips align to stitch marks

### 4. Optimization

**Coverage Mode (DP):**
- Uses dynamic programming with backpointers
- Maximizes total score while minimizing gaps
- Enforces tail coverage (clips must reach near video end)

**Highlights Mode (Greedy):**
- Generates all candidates, scores them
- Greedily selects top-k non-overlapping clips

### 5. Review Loop

Two-stage review for quality assurance:

**Heuristic Checks (free):**
- Duration within bounds
- No forbidden stitch marks inside
- Speech within first 500ms (hook)
- Silence ratio below threshold
- Speech density at cut points

**LLM Review (if borderline):**
- Uses structured outputs (JSON schema)
- Can only return PASS/FAIL
- Can suggest shifting to adjacent boundaries
- Cannot suggest arbitrary timestamps

### 6. Render with Smooth Transitions

- Preview: Low-res (480p) for quick review
- Final: Full quality with audio/video fades
  - **300ms fade-in**: Smooth entry to each clip
  - **400ms fade-out**: Gradual exit prevents jarring cuts

## Performance

- **Boundary detection**: ~1x realtime on CPU
- **Optimization**: O(n·k) for candidates, O(n) for DP
- **LLM review**: ~1-2s per clip (only for borderline clips)
- **Render**: Depends on video length and quality settings

### Example Results

Tested on a 722-second kids' ABC song video:
- **15 clips** generated
- **98.7% coverage** of original video
- Smooth transitions with audio/video fades

## Troubleshooting

### "ffmpeg not found"

Install ffmpeg:
```bash
# macOS
brew install ffmpeg

# Ubuntu
sudo apt install ffmpeg
```

### "CUDA out of memory"

Use CPU mode:
```yaml
# In config/pipeline_config.yaml
detectors:
  whisper:
    device: cpu
```

### "No valid clips found"

- Check if video has audio
- Try reducing `min_clip_duration`
- Check if stitch marks are blocking all candidates

### LLM review failing

- Ensure `OPENAI_API_KEY` is set
- Try `--no-llm` to use heuristics only

### Abrupt cuts

- Ensure `fade_in_ms` and `fade_out_ms` are configured
- Increase merge threshold in pipeline config
- Enable energy detector for better cut point detection

## License

MIT License
