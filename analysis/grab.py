"""Frame capture from the live stream.

yt-dlp resolves the YouTube page to a direct HLS URL (valid for hours),
then ffmpeg pulls single JPEG frames from it. No video is stored.
"""

import subprocess
import tempfile
import time

import cv2
import numpy as np

import config


class StreamUnavailable(Exception):
    pass


_cached_url: str | None = None
_cached_at: float = 0.0


def resolve_stream_url(force: bool = False) -> str:
    """Return a direct HLS URL for the live stream, cached until stale."""
    global _cached_url, _cached_at
    if (
        not force
        and _cached_url
        and time.monotonic() - _cached_at < config.STREAM_URL_TTL_SECONDS
    ):
        return _cached_url

    result = subprocess.run(
        [
            "yt-dlp",
            "-g",
            "-f",
            f"best[height<={config.GRAB_MAX_HEIGHT}]",
            config.STREAM_URL,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise StreamUnavailable(f"yt-dlp failed: {result.stderr.strip()[:500]}")

    _cached_url = result.stdout.strip().splitlines()[0]
    _cached_at = time.monotonic()
    return _cached_url


def grab_frame(hls_url: str) -> np.ndarray:
    """Grab one frame as a BGR numpy array via ffmpeg piping JPEG to stdout."""
    result = subprocess.run(
        [
            "ffmpeg",
            "-loglevel", "error",
            "-i", hls_url,
            "-frames:v", "1",
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "-q:v", "2",
            "-",
        ],
        capture_output=True,
        timeout=45,
    )
    if result.returncode != 0 or not result.stdout:
        raise StreamUnavailable(f"ffmpeg failed: {result.stderr.decode()[:500]}")

    frame = cv2.imdecode(np.frombuffer(result.stdout, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise StreamUnavailable("ffmpeg produced undecodable output")
    return frame


def _grab_two(hls_url: str, gap_seconds: float):
    """Grab two frames `gap_seconds` apart from ONE ffmpeg session.

    Two separate invocations can return the identical frame because live
    HLS serves ~5s segments and each fresh connection starts at a segment
    boundary. Decoding continuously and sampling with the fps filter
    guarantees the frames really are `gap_seconds` apart.
    """
    with tempfile.TemporaryDirectory() as tmp:
        pattern = f"{tmp}/f%d.jpg"
        result = subprocess.run(
            [
                "ffmpeg",
                "-loglevel", "error",
                "-i", hls_url,
                "-vf", f"fps=1/{gap_seconds}",
                "-frames:v", "2",
                "-q:v", "2",
                pattern,
            ],
            capture_output=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise StreamUnavailable(f"ffmpeg failed: {result.stderr.decode()[:500]}")
        frames = [cv2.imread(f"{tmp}/f{i}.jpg") for i in (1, 2)]
    if any(f is None for f in frames):
        raise StreamUnavailable("ffmpeg did not produce two frames")
    return frames[0], frames[1]


def grab_pair(gap_seconds: float = config.MOTION_FRAME_GAP_SECONDS):
    """Grab two frames `gap_seconds` apart; retries URL resolution once."""
    try:
        url = resolve_stream_url()
        return _grab_two(url, gap_seconds)
    except StreamUnavailable:
        url = resolve_stream_url(force=True)
        return _grab_two(url, gap_seconds)


if __name__ == "__main__":
    a, b = grab_pair()
    cv2.imwrite("frame_a.jpg", a)
    cv2.imwrite("frame_b.jpg", b)
    print(f"grabbed {a.shape} and {b.shape} -> frame_a.jpg, frame_b.jpg")
