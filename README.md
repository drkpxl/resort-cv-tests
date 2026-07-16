# Resort CV Tests

A learning project: computer vision on the [Winter Park gondola live cam](https://www.youtube.com/watch?v=vyWAzJAWv4w).
A Python service samples the YouTube live stream and answers three questions on a web dashboard:

1. **Is the gondola running?** — classical CV: frame differencing inside a region drawn over the cable span. No ML.
2. **How long is the lift line?** — YOLOv8-nano (local, ~6MB) detects people; only those whose feet land inside a queue polygon count. Current count + daily cumulative sightings + daily peak.
3. **What's the weather?** — sunny / rainy / snow / night / twilight from brightness, contrast, saturation, and ground-whiteness stats, gated by computed sun elevation (dusk light is statistically identical to overcast — astronomy breaks the tie).

There is also a **real-time tracking mode** (4 fps continuous decode, ByteTrack IDs with motion trails, unique-queue-visitor counting) streamed to the browser as MJPEG.

Everything runs locally. No cloud APIs, no LLMs, no video stored — single frames only.

## Prerequisites

- Python managed by [uv](https://docs.astral.sh/uv/)
- `ffmpeg` and `yt-dlp` on PATH (`brew install ffmpeg yt-dlp`)

## Run

```sh
uv sync
uv run uvicorn server:app --port 8500
```

- Dashboard: http://localhost:8500 — embedded live video, status cards, 2-hour timelines. Mobile-friendly.
- Debug view: http://localhost:8500/static/debug.html — live tracking feed (click Start; one viewer at a time) plus the latest analyzed snapshot with all CV work drawn on it.

The analyzer samples the stream every 15s and stores observations in `data.db` (SQLite, created on first run). YOLO weights download automatically on first run.

## Layout

```
server.py              FastAPI app, 15s analyzer loop, /api/status, /api/live-feed
config.py              ALL tunables: ROIs (fractional coords), thresholds, cadence, location
analysis/
  grab.py              yt-dlp URL resolution + ffmpeg frame capture
  motion.py            frame differencing in the cable ROI
  line.py              YOLO person count filtered by queue polygon
  weather.py           brightness/contrast/color stats + solar elevation gate
  realtime.py          4fps tracking pipeline, jitter buffer, MJPEG generator
  tracker_gondola.yaml ByteTrack config tuned for tiny (~20px) people
  annotate.py          draws ROIs/boxes/stats on the debug frame
static/                dashboard (Pico.css, plain JS)
```

## Tuning

Everything tunable lives in `config.py` with comments. The debug view is the tuning tool:
ROIs are fractions of frame size, so adjust `MOTION_ROI` / `QUEUE_POLYGON` until the overlays
sit where you want them. Raw stats (motion score, brightness, contrast, saturation) are stored
with every observation so thresholds can be re-fit against history later.

## Lessons learned the hard way (a.k.a. why some code looks the way it does)

- **YOLO found zero people at first.** Default 640px input shrinks distant people below detectability; run at the stream's native width (`YOLO_IMAGE_SIZE = 1280`).
- **Two ffmpeg grabs can return the identical frame** on live HLS (~5s segments), which read as "not moving". Both motion frames now come from one continuous decode session.
- **The tracker assigned no IDs** despite good detections: stock ByteTrack won't birth tracks below 0.6 confidence, and tiny people score 0.25–0.5. See `tracker_gondola.yaml`.
- **The live feed stuttered** — burst-freeze-burst. HLS delivers ~5s chunks; a jitter buffer (reader thread + steady 4fps playout) smooths it, buying smoothness with latency like every video player.
- **Leaked ffmpeg pipelines killed everything.** A stalled MJPEG generator that never yields can't be cleaned up on disconnect; enough zombie pipelines and YouTube stops serving the client at all. Fixed with a singleton pipeline (new stream kills the old) and a 20s stall watchdog.
- **Sunset classified as rainy.** Flat dusk light and overcast are the same to pixel statistics, and auto-exposure hides the darkness. Fixed by computing sun elevation from lat/lon + clock (NOAA approximation in `weather.py`) — a tiny example of sensor fusion.

## Honest limitations

- Weather thresholds were calibrated on one sunny July afternoon; rainy/snow need real examples to tune against.
- "Sightings" ≠ unique people: the 15s loop can't tell people apart (one person lingering counts repeatedly). The real-time tracker counts unique IDs but IDs churn on occlusion, so it overcounts somewhat.
- The queue polygon is a summer guess at where the winter line forms; adjust when the snow arrives.
