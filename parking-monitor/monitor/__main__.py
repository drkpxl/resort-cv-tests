"""Entry point.

  uv run python -m monitor            # scheduler + web server (default)
  uv run python -m monitor --once     # run one cycle, print the row, exit
  uv run python -m monitor --no-web   # scheduler only (no web server)
"""
import sys
import threading

import config
from . import db, pipeline, scheduler


def main() -> None:
    args = set(sys.argv[1:])
    db.init()

    if "--once" in args:
        row = pipeline.run_once()
        print(row)
        return

    if "--no-web" in args:
        scheduler.run_forever()
        return

    threading.Thread(target=scheduler.run_forever, daemon=True).start()
    import uvicorn
    from .web import app
    print(f"web UI on http://0.0.0.0:{config.WEB_PORT}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=config.WEB_PORT)


if __name__ == "__main__":
    main()
