"""FastAPI app: background analyzer loop + SQLite history + status API."""

import asyncio
import logging
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import config
from analysis import annotate, grab, line, motion, weather

log = logging.getLogger("gondola")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
  id INTEGER PRIMARY KEY,
  ts TEXT NOT NULL,
  gondola_moving INTEGER,
  motion_score REAL,
  person_count INTEGER,
  line_status TEXT,
  weather TEXT,
  brightness REAL,
  contrast REAL,
  saturation REAL,
  white_fraction REAL
);
CREATE INDEX IF NOT EXISTS idx_observations_ts ON observations(ts);
"""


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def analyze_once() -> dict:
    """One full analysis cycle. Blocking; run in a thread from the loop."""
    frame_a, frame_b = grab.grab_pair()
    m = motion.analyze(frame_a, frame_b)
    l = line.analyze(frame_b)
    w = weather.analyze(frame_b)

    debug = annotate.annotate(frame_b, m, l, w)
    Path(config.DEBUG_FRAME_PATH).parent.mkdir(exist_ok=True)
    cv2.imwrite(config.DEBUG_FRAME_PATH, debug)

    row = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gondola_moving": int(m["moving"]),
        "motion_score": m["motion_score"],
        "person_count": l["person_count"],
        "line_status": l["line_status"],
        "weather": w["weather"],
        "brightness": w["brightness"],
        "contrast": w["contrast"],
        "saturation": w["saturation"],
        "white_fraction": w["white_fraction"],
    }
    with db() as conn:
        conn.execute(
            """INSERT INTO observations
               (ts, gondola_moving, motion_score, person_count, line_status,
                weather, brightness, contrast, saturation, white_fraction)
               VALUES (:ts, :gondola_moving, :motion_score, :person_count,
                       :line_status, :weather, :brightness, :contrast,
                       :saturation, :white_fraction)""",
            row,
        )
    return row


async def analyzer_loop():
    while True:
        try:
            row = await asyncio.to_thread(analyze_once)
            log.info(
                "moving=%s queue=%s(%s) weather=%s",
                bool(row["gondola_moving"]), row["person_count"],
                row["line_status"], row["weather"],
            )
        except grab.StreamUnavailable as exc:
            log.warning("stream unavailable: %s", exc)
        except Exception:
            log.exception("analysis cycle failed")
        await asyncio.sleep(config.ANALYZE_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    with db() as conn:
        conn.executescript(SCHEMA)
    task = asyncio.create_task(analyzer_loop())
    yield
    task.cancel()


app = FastAPI(title="Gondola Cam CV", lifespan=lifespan)


@app.get("/api/status")
def status():
    since = (
        datetime.now(timezone.utc) - timedelta(hours=config.HISTORY_HOURS)
    ).isoformat(timespec="seconds")
    # "Today" starts at local midnight (observations are stored in UTC)
    midnight = datetime.now().astimezone().replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    since_midnight = midnight.astimezone(timezone.utc).isoformat(timespec="seconds")
    with db() as conn:
        rows = [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM observations WHERE ts >= ? ORDER BY ts", (since,)
            )
        ]
        day = dict(
            conn.execute(
                """SELECT COALESCE(SUM(person_count), 0) AS total_sightings,
                          COALESCE(MAX(person_count), 0) AS peak
                   FROM observations WHERE ts >= ?""",
                (since_midnight,),
            ).fetchone()
        )
        day["peak_ts"] = None
        if day["peak"] > 0:
            day["peak_ts"] = conn.execute(
                "SELECT ts FROM observations WHERE ts >= ? AND person_count = ?"
                " ORDER BY ts LIMIT 1",
                (since_midnight, day["peak"]),
            ).fetchone()["ts"]
    latest = rows[-1] if rows else None
    stale = True
    if latest:
        age = datetime.now(timezone.utc) - datetime.fromisoformat(latest["ts"])
        stale = age.total_seconds() > config.STALE_AFTER_SECONDS
    return JSONResponse(
        {
            "latest": latest,
            "stale": stale,
            "history": rows,
            "today": day,
            "video_id": config.YOUTUBE_VIDEO_ID,
            "interval_seconds": config.ANALYZE_INTERVAL_SECONDS,
        }
    )


@app.get("/")
def index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
