"""One monitoring cycle: snapshot -> detect -> LLM -> % full -> store."""
import datetime
import os

import cv2
from PIL import Image

import config
from . import db, detect, grab, llm


def _dirs():
    img = os.path.join(config.DATA_DIR, "images")
    thumb = os.path.join(config.DATA_DIR, "thumbs")
    os.makedirs(img, exist_ok=True)
    os.makedirs(thumb, exist_ok=True)
    return img, thumb


def run_once() -> dict:
    """Run one full cycle and persist a row. Raises if grab/detect fails
    (no partial row); an LLM failure is soft (llm_count stays None)."""
    db.init()
    img_dir, thumb_dir = _dirs()
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # 1. snapshot
    frame = grab.grab_frame()
    full_path = os.path.join(img_dir, f"{ts}.jpg")
    cv2.imwrite(full_path, frame)

    # 2. vision floor + overlay
    vision_count, coverage_pct, overlay = detect.detect(frame)
    cv2.imwrite(os.path.join(img_dir, f"{ts}_floor.jpg"), overlay)

    # 3. thumbnail of the full frame
    im = Image.open(full_path)
    im.thumbnail((320, 320))
    im.save(os.path.join(thumb_dir, f"{ts}.jpg"), "JPEG", quality=80)

    # 4. LLM add-misses (fail soft)
    try:
        llm_count = llm.add_misses(overlay, vision_count)
    except Exception as e:  # noqa: BLE001
        print(f"[{ts}] LLM step failed: {e}")
        llm_count = None

    # 5. % full (both signals uncalibrated)
    over_cap = round(llm_count / config.CAPACITY * 100, 1) if llm_count else None

    row = {
        "ts": ts,
        "image_url": f"/images/{ts}.jpg",
        "thumb_url": f"/thumbs/{ts}.jpg",
        "vision_count": vision_count,
        "llm_count": llm_count,
        "percent_full": round(coverage_pct, 1),
        "count_over_capacity": over_cap,
        "source_url": config.YOUTUBE_URL,
    }
    db.insert(row)
    _prune_old(img_dir, thumb_dir)
    return row


def _prune_old(img_dir: str, thumb_dir: str) -> None:
    """Delete image/thumb files older than IMAGE_RETENTION_DAYS (DB rows stay)."""
    days = config.IMAGE_RETENTION_DAYS
    if not days:
        return
    import time
    cutoff = time.time() - days * 86400
    for d in (img_dir, thumb_dir):
        for name in os.listdir(d):
            p = os.path.join(d, name)
            try:
                if os.path.isfile(p) and os.path.getmtime(p) < cutoff:
                    os.remove(p)
            except OSError:
                pass
