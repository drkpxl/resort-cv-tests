"""FastAPI web UI: a table of recent observations with thumbnails."""
import datetime
import html
import os

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import config
from . import db

app = FastAPI(title="Parking monitor")

# Static mounts (dirs are created by the pipeline / __main__ before startup).
for _sub in ("images", "thumbs"):
    os.makedirs(os.path.join(config.DATA_DIR, _sub), exist_ok=True)
app.mount("/images", StaticFiles(directory=os.path.join(config.DATA_DIR, "images")),
          name="images")
app.mount("/thumbs", StaticFiles(directory=os.path.join(config.DATA_DIR, "thumbs")),
          name="thumbs")

_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Parking monitor</title><style>
:root{{--bg:#0f120f;--surface:#161a16;--ink:#e7e9e2;--muted:#8f958a;--hair:#282d27;--accent:#e5853b;
--mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;--sans:"Helvetica Neue",Arial,system-ui,sans-serif;}}
@media (prefers-color-scheme:light){{:root{{--bg:#e6e7e1;--surface:#f0f1eb;--ink:#181b18;--muted:#5e645c;--hair:#d3d5ce;--accent:#bc5e19;}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);}}
main{{max-width:1000px;margin:0 auto;padding:1.4rem 1rem 3rem}}
h1{{font-family:var(--mono);font-weight:600;letter-spacing:-.02em;font-size:1.5rem;margin:0}}
.sub{{color:var(--muted);font-size:.85rem;margin:.3rem 0 1.4rem}}
.sub b{{color:var(--accent)}}
table{{width:100%;border-collapse:collapse;font-size:.9rem}}
th,td{{padding:.5rem .6rem;border-bottom:1px solid var(--hair);text-align:right;vertical-align:middle}}
th:first-child,td:first-child,th.l,td.l{{text-align:left}}
th{{font-family:var(--mono);font-weight:600;font-size:.66rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}}
td.num{{font-family:var(--mono);font-variant-numeric:tabular-nums;font-size:1.05rem}}
td.llm{{color:var(--accent)}}
img{{display:block;width:150px;border:1px solid var(--hair);border-radius:4px}}
.t{{font-family:var(--mono);font-size:.78rem}}.rel{{color:var(--muted);font-size:.72rem}}
.tag{{font-family:var(--mono);font-size:.6rem;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}}
.empty{{color:var(--muted);padding:2rem 0}}
</style></head><body><main>
<h1>Brighton parking monitor</h1>
<p class="sub">Vehicle counts every {interval} min · <b>vision</b> = detector floor ·
<b>LLM</b> = floor + missed (gemma4:e4b) · % full is <b>uncalibrated</b>.</p>
{table}
</main></body></html>"""


def _rel(ts: str) -> str:
    try:
        t = datetime.datetime.strptime(ts, "%Y%m%dT%H%M%SZ").replace(
            tzinfo=datetime.timezone.utc)
    except ValueError:
        return ""
    secs = (datetime.datetime.now(datetime.timezone.utc) - t).total_seconds()
    if secs < 90:
        return "just now"
    if secs < 5400:
        return f"{int(secs // 60)} min ago"
    if secs < 172800:
        return f"{int(secs // 3600)} h ago"
    return f"{int(secs // 86400)} d ago"


def _fmt_local(ts: str) -> str:
    try:
        t = datetime.datetime.strptime(ts, "%Y%m%dT%H%M%SZ").replace(
            tzinfo=datetime.timezone.utc).astimezone()
        return t.strftime("%b %d %H:%M")
    except ValueError:
        return ts


def _table(rows: list[dict]) -> str:
    if not rows:
        return '<p class="empty">No observations yet — the first cycle runs shortly.</p>'
    head = ('<tr><th class="l">Snapshot</th><th class="l">Time</th>'
            '<th>Vision</th><th>LLM</th><th>% full</th></tr>')
    body = ""
    for r in rows:
        img = html.escape(r["image_url"])
        thumb = html.escape(r["thumb_url"])
        llm = r["llm_count"] if r["llm_count"] is not None else "—"
        pct = f'{r["percent_full"]:.0f}%' if r["percent_full"] is not None else "—"
        body += (
            f'<tr><td class="l"><a href="{img}" target="_blank" rel="noopener">'
            f'<img src="{thumb}" alt="lot snapshot" loading="lazy"></a></td>'
            f'<td class="l"><div class="t">{_fmt_local(r["ts"])}</div>'
            f'<div class="rel">{_rel(r["ts"])}</div></td>'
            f'<td class="num">{r["vision_count"]}</td>'
            f'<td class="num llm">{llm}</td>'
            f'<td class="num">{pct}<div class="tag">uncal.</div></td></tr>'
        )
    return f"<table><thead>{head}</thead><tbody>{body}</tbody></table>"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    rows = db.recent(config.WEB_ROW_LIMIT)
    return _PAGE.format(interval=config.INTERVAL_MINUTES, table=_table(rows))
