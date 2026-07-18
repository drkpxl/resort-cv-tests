"""LLM add-misses step: a cheap local vision model tops up the detector floor.

Sends the detector overlay to Ollama (gemma4:e4b, thinking OFF) and asks it to
count the vehicles the detector missed, on top of the floor N.
"""
import base64
import json
import re
import urllib.request

import cv2

import config


class LLMError(Exception):
    pass


_PROMPT = (
    "An automatic vehicle detector has marked N = {n} vehicles in this parking-lot "
    "image, shown as red silhouettes each with a green dot at its base. Treat {n} as "
    "a reliable FLOOR: the marked vehicles are real, but the detector MISSES vehicles "
    "that are distant, tightly packed, or partly hidden. Find the vehicles it MISSED "
    "(unmarked cars, trucks, vans, SUVs), count them, and add to the floor. "
    'Reply ONLY JSON {{"detector_floor":{n},"missed":<int>,"total":<int>}}.'
)


def add_misses(overlay_bgr, floor_n: int) -> int:
    """Return the model's total count (floor + misses). Retries once, then raises."""
    ok, buf = cv2.imencode(".jpg", overlay_bgr)
    img = base64.b64encode(buf.tobytes()).decode()
    body = json.dumps({
        "model": config.LLM_MODEL,
        "stream": False,
        "format": "json",
        "think": config.LLM_THINK,
        "keep_alive": config.LLM_KEEP_ALIVE,
        "options": {"temperature": 0},
        "messages": [{"role": "user", "content": _PROMPT.format(n=floor_n),
                      "images": [img]}],
    }).encode()

    last = ""
    for attempt in range(2):
        try:
            req = urllib.request.Request(
                config.OLLAMA_HOST + "/api/chat", body,
                {"Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=config.LLM_TIMEOUT))
            txt = r["message"]["content"].strip()
            m = re.search(r"\{.*\}", txt, re.S)
            if not m:
                last = f"no JSON in reply: {txt[:80]!r}"
                continue
            total = json.loads(m.group(0)).get("total")
            if total is None:
                last = f"no 'total' field: {txt[:80]!r}"
                continue
            return int(total)
        except Exception as e:  # noqa: BLE001 - fail soft, report last error
            last = repr(e)[:150]
    raise LLMError(last)
