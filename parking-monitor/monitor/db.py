"""SQLite storage for observations."""
import os
import sqlite3

import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    id                  INTEGER PRIMARY KEY,
    ts                  TEXT,      -- UTC ISO-8601
    image_url           TEXT,      -- web route for the full frame
    thumb_url           TEXT,
    vision_count        INTEGER,   -- detector floor
    llm_count           INTEGER,   -- floor + model additions (NULL on failure)
    percent_full        REAL,      -- perspective coverage % (uncalibrated)
    count_over_capacity REAL,      -- llm_count / capacity * 100 (uncalibrated)
    source_url          TEXT
)
"""


def _path() -> str:
    return os.path.join(config.DATA_DIR, "monitor.db")


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(_path())
    con.row_factory = sqlite3.Row
    return con


def init() -> None:
    os.makedirs(config.DATA_DIR, exist_ok=True)
    con = _connect()
    con.execute(_SCHEMA)
    con.commit()
    con.close()


def insert(row: dict) -> None:
    con = _connect()
    con.execute(
        """INSERT INTO observations
           (ts, image_url, thumb_url, vision_count, llm_count,
            percent_full, count_over_capacity, source_url)
           VALUES (:ts, :image_url, :thumb_url, :vision_count, :llm_count,
                   :percent_full, :count_over_capacity, :source_url)""",
        row,
    )
    con.commit()
    con.close()


def recent(limit: int) -> list[dict]:
    con = _connect()
    rows = con.execute(
        "SELECT * FROM observations ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]
