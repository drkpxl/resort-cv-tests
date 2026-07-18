# Build prompt — Parking-lot occupancy monitor

Build a small, standalone service that watches a public parking-lot webcam and
logs how full it is over time. This is a fresh project — do **not** reuse or
depend on any "gondola" / resort-CV code. Prefer **uv** for everything
(`uv init`, `uv add`, `uv run`).

## What it does (one cycle, every 15 minutes)

1. **Snapshot.** Grab one still frame from a YouTube live feed.
   Resolve the stream with `yt-dlp -g -f "best[height<=720]" <URL>` and pull a
   single frame with `ffmpeg -i <hls_url> -frames:v 1 -q:v 2 out.jpg`. Save the
   full-resolution image to disk under `data/images/<UTC-timestamp>.jpg`.
2. **Vision model (the "floor").** Run a high-recall object detector to count
   the vehicles it is confident about. Use Ultralytics `yolov8x-seg` at
   `imgsz=1280` with **SAHI-style tiling**: slice the frame into an overlapping
   grid (start 4 rows × 5 cols, 25% overlap), run the model on each crop, map
   boxes back to full-frame coords, and merge with greedy NMS (IoU ~0.45). Count
   only vehicles whose footprint (bottom-center of box) falls inside a
   configurable **lot polygon** (fractional coords). Produce two things:
   - `vision_count` = the merged detection count.
   - an **overlay image**: the frame with each detected vehicle's silhouette
     tinted red and a green dot at its base — **no lot box, no text**.
3. **LLM model ("add the misses").** Send the *overlay* image to a local Ollama
   vision model and ask it to add what the detector missed. **Use
   `gemma4:e4b` with thinking OFF** (`"think": false`) — thinking mode is slower
   and less accurate for this task, and small models can hang with it on.
   Request format `json`, `keep_alive` configurable. Prompt:

   > An automatic vehicle detector has marked N = {vision_count} vehicles in this
   > parking-lot image, shown as red silhouettes each with a green dot at its
   > base. Treat {vision_count} as a reliable FLOOR: the marked vehicles are real,
   > but the detector MISSES vehicles that are distant, tightly packed, or partly
   > hidden. Find the vehicles it MISSED, count them, and add to the floor.
   > Reply ONLY JSON {"detector_floor":N,"missed":<int>,"total":<int>}.

   `llm_count` = `total` from the response (parse leniently; fall back to a regex
   `\{.*\}` extract).
4. **Percent full.** Compute a perspective-corrected surface-coverage estimate:
   a homography maps the lot polygon to a top-down rectangle; weight each vehicle
   pixel by the homography's local area scale (`1/w³`) and divide by the weighted
   lot area. Store that as `percent_full`. Also store `count_over_capacity`
   (`llm_count / capacity`) as a secondary. **Flag % full as uncalibrated in the
   UI** — it can't be tuned until we capture a 0%-empty and 100%-full frame for
   this camera.
5. **Persist.** Append one row to a SQLite DB (`data/monitor.db`), table
   `observations`:
   `id INTEGER PK`, `ts TEXT` (UTC ISO-8601), `image_path TEXT`,
   `image_url TEXT` (the web route that serves the full image),
   `vision_count INTEGER`, `llm_count INTEGER`, `percent_full REAL`,
   `count_over_capacity REAL`, `source_url TEXT`. Also write a small thumbnail
   (e.g. 320px wide) next to the full image for the table view.

## The website

Serve a basic web UI (FastAPI + uvicorn, or stdlib if you prefer minimal). One
page, mobile-responsive:

- A **table of recent observations, newest on top**. Each row: the **thumbnail**,
  the timestamp (local + relative), `vision_count`, `llm_count`, and
  `percent_full` (labeled "uncalibrated").
- **Tapping the thumbnail opens the full image in a new browser tab/window**
  (link the thumbnail to `image_url`, `target="_blank"`).
- Serve full images and thumbnails as static files.
- Keep it clean and readable; no framework needed on the frontend.

## Scheduling & structure

- A simple scheduler: a loop that runs the cycle, then sleeps to the next
  15-minute boundary (or APScheduler). One process can run both the scheduler
  and the web server (background thread/task), or split them — your call.
- Config in one file (`config.py` or `.toml`): `YOUTUBE_URL`, `LOT_POLYGON`
  (fractional points), `CAPACITY`, `INTERVAL_MINUTES=15`, `OLLAMA_HOST`
  (default `http://localhost:11434`), `VISION_MODEL="yolov8x-seg"`,
  `LLM_MODEL="gemma4:e4b"`, tile grid + overlap.
- Dependencies via uv: `ultralytics`, `opencv-python`, `pillow`, `fastapi`,
  `uvicorn`, `httpx` (or stdlib urllib). `ffmpeg` and `yt-dlp` on PATH.
- Fail soft: if a snapshot or a model call fails, log it and skip that cycle;
  never crash the loop. Don't store partial rows.

## Notes / gotchas we already hit

- **Thinking off, always** (`"think": false`) — faster and more accurate here.
- **Ollama Cloud serializes to one call at a time**; a local model doesn't, but
  don't stack concurrent calls at the same model.
- The detector floor is the accuracy bottleneck — the LLM only adds ~10% on top.
  If counts look low, raise detector recall (finer tiling / lower conf) before
  blaming the LLM.
- Keep the full image; the thumbnail is only for the table.

## Deliverable

A `uv`-managed project that runs with something like
`uv run python -m monitor` (starts scheduler + web server on a configurable
port), writing to `data/monitor.db` and `data/images/`, with the table UI live.
