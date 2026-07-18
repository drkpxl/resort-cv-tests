"""Grab one frame from the YouTube live feed (yt-dlp resolves, ffmpeg captures)."""
import subprocess

import cv2
import numpy as np

import config


class GrabError(Exception):
    pass


def grab_frame() -> np.ndarray:
    """Return one BGR frame from the live feed. Raises GrabError on failure."""
    r = subprocess.run(
        ["yt-dlp", "-g", "-f", f"best[height<={config.GRAB_MAX_HEIGHT}]",
         config.YOUTUBE_URL],
        capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0 or not r.stdout.strip():
        raise GrabError(f"yt-dlp failed: {r.stderr.strip()[:200]}")
    hls = r.stdout.strip().splitlines()[0]

    out = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", hls, "-frames:v", "1",
         "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "2", "-"],
        capture_output=True, timeout=60,
    )
    if out.returncode != 0 or not out.stdout:
        raise GrabError(f"ffmpeg failed: {out.stderr.decode()[:200]}")

    frame = cv2.imdecode(np.frombuffer(out.stdout, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise GrabError("ffmpeg produced an undecodable frame")
    return frame
