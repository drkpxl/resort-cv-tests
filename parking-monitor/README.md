# Parking monitor

A standalone service that watches a parking-lot webcam and logs how full it is.
Every 15 minutes it snapshots the YouTube feed, runs a high-recall YOLO detector
(the "floor"), asks a local vision model (`gemma4:e4b`, thinking off) to add the
vehicles the detector missed, stores the result in SQLite, and serves a simple
table website with thumbnails.

Runs 100% on one machine. No cloud, no external APIs.

## Prerequisites (on the box)

- [uv](https://docs.astral.sh/uv/)
- `ffmpeg` and `yt-dlp` on PATH — e.g. `brew install ffmpeg yt-dlp`
- [Ollama](https://ollama.com) running with the vision model:
  `ollama pull gemma4:e4b`

## Setup

```sh
cd parking-monitor
uv sync                 # installs ultralytics, fastapi, uvicorn, pillow
```

The first detector run downloads the `yolov8x-seg` weights (~140 MB) automatically.

## Run

```sh
uv run python -m monitor --once    # one cycle, prints the row (test)
uv run python -m monitor           # scheduler + web UI (the service)
```

Then open `http://<box-ip>:8600/` — newest snapshots on top; tap a thumbnail to
open the full image.

Keep it running with tmux / nohup / launchd, e.g.:

```sh
nohup uv run python -m monitor > monitor.log 2>&1 &
```

## Configuration

Everything lives in `config.py`: the YouTube URL, the lot polygon (fractional
coords — tune per camera against a snapshot), capacity, the 15-minute interval,
the models, the tiling grid (`TILES`/`OVERLAP` — raise for more detector recall),
the Ollama host, and the web port.

## What the columns mean

- **Vision** — the detector floor (vehicles YOLO is confident about).
- **LLM** — floor + the misses the model found. This is the count to trust.
- **% full** — perspective-corrected surface coverage. **Uncalibrated** until we
  capture a 0%-empty and a 100%-full reference frame for this camera; treat it as
  a relative trend, not an absolute.

## Notes

- Thinking is kept **off** on the LLM (`LLM_THINK = False`) — it's faster and more
  accurate for counting; small models can hang with it on.
- The loop is fail-soft: a failed snapshot/detect skips the cycle; a failed LLM
  call stores the row with `llm_count = NULL`. It never crashes.
- Accuracy is bottlenecked by the detector's recall, not the LLM — if counts look
  low, raise `TILES` / lower `CONF` in `config.py`.
