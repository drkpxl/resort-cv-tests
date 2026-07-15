"""Real-time person tracking over a continuous decode of the live stream.

Where the 15-second dashboard loop answers "what is the state right now",
this mode processes every frame (well, 4 per second) so consecutive
detections overlap and can be stitched into TRACKS: the same person keeps
the same ID from frame to frame. That's what makes unique-visitor counting
possible - and watching the IDs swap when people cross paths teaches you
exactly why tracking is hard.

ffmpeg decodes the HLS stream to raw BGR frames on a pipe; YOLO runs with
its built-in BoT-SORT tracker (persist=True keeps tracker state between
calls). Annotated frames stream to the browser as MJPEG.
"""

import subprocess
import time
from collections import defaultdict, deque

import cv2
import numpy as np

import config
from analysis import grab
from analysis.line import _polygon_pixels

_model = None


def _get_model():
    # Separate instance from line.py: track() keeps tracker state on the
    # model object, and the dashboard loop's predict() calls would share it.
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO("yolov8n.pt")
    return _model


def _open_stream() -> subprocess.Popen:
    url = grab.resolve_stream_url()
    return subprocess.Popen(
        [
            "ffmpeg",
            "-loglevel", "error",
            "-i", url,
            "-vf",
            f"fps={config.REALTIME_FPS},"
            f"scale={config.REALTIME_WIDTH}:{config.REALTIME_HEIGHT}",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )


def mjpeg_stream():
    """Generator yielding multipart JPEG chunks of annotated live frames."""
    model = _get_model()
    proc = _open_stream()
    w, h = config.REALTIME_WIDTH, config.REALTIME_HEIGHT
    frame_bytes = w * h * 3
    polygon = _polygon_pixels((h, w))

    trails = defaultdict(lambda: deque(maxlen=config.TRAIL_LENGTH))
    queue_ids: set[int] = set()   # every track ID ever seen inside the queue
    last_time = time.monotonic()
    fps = 0.0

    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if buf is None or len(buf) < frame_bytes:
                break
            frame = np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()

            results = model.track(
                frame,
                persist=True,
                conf=config.PERSON_CONFIDENCE,
                classes=[0],
                imgsz=config.YOLO_IMAGE_SIZE,
                tracker="analysis/tracker_gondola.yaml",
                verbose=False,
            )
            boxes = results[0].boxes

            in_queue_now = 0
            people_now = 0
            if boxes.id is not None:
                people_now = len(boxes.id)
                for xyxy, tid in zip(boxes.xyxy, boxes.id.int().tolist()):
                    x1, y1, x2, y2 = map(int, xyxy.tolist())
                    feet = ((x1 + x2) // 2, y2)
                    trails[tid].append(feet)
                    inside = cv2.pointPolygonTest(polygon, feet, False) >= 0
                    if inside:
                        in_queue_now += 1
                        queue_ids.add(tid)
                    color = (0, 255, 0) if inside else (180, 180, 180)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"#{tid}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    if len(trails[tid]) > 1:
                        pts = np.array(trails[tid], np.int32)
                        cv2.polylines(frame, [pts], False, color, 1)

            cv2.polylines(frame, [polygon], True, (0, 255, 255), 2)

            now = time.monotonic()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_time, 1e-6))
            last_time = now
            banner = (
                f"tracking {people_now} people | in queue now: {in_queue_now} | "
                f"unique queue visitors this session: {len(queue_ids)} | "
                f"{fps:.1f} fps processed"
            )
            cv2.rectangle(frame, (0, h - 28), (w, h), (0, 0, 0), -1)
            cv2.putText(frame, banner, (8, h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + jpg.tobytes() + b"\r\n")
    finally:
        proc.kill()
