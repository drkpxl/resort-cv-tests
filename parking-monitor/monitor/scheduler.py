"""Run the pipeline on a fixed wall-clock cadence, fail-soft forever."""
import time

import config
from . import pipeline


def _sleep_to_next_boundary() -> float:
    interval = config.INTERVAL_MINUTES * 60
    now = time.time()
    return max(1.0, (now // interval + 1) * interval - now)


def run_forever() -> None:
    while True:
        try:
            row = pipeline.run_once()
            print(f"[{row['ts']}] logged  vision={row['vision_count']} "
                  f"llm={row['llm_count']}  cover={row['percent_full']}%", flush=True)
        except Exception as e:  # noqa: BLE001 - never crash the loop
            print(f"cycle failed: {repr(e)[:200]}", flush=True)
        time.sleep(_sleep_to_next_boundary())
