"""LLM add-misses step: a cheap local vision model tops up the detector floor.

Sends the detector overlay to oMLX (unsloth-gemma-4-E4B-it-qat-oQ4, thinking OFF)
and asks it to count the vehicles the detector missed, on top of the floor N.
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


def _extract_json(text: str):
    """Return the first JSON object found in text, or None."""
    text = text.strip()
    # Try whole text first.
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    # Then look for a JSON object anywhere in the response.
    for match in re.finditer(r"\{.*?\}", text, re.S):
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
    return None


def add_misses(overlay_bgr, floor_n: int) -> int:
    """Return the model's total count (floor + misses). Retries once, then raises."""
    ok, buf = cv2.imencode(".jpg", overlay_bgr)
    if not ok:
        raise LLMError("failed to encode overlay image")
    img = base64.b64encode(buf.tobytes()).decode()

    body = json.dumps({
        "model": config.LLM_MODEL,
        "stream": False,
        "temperature": 0,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": _PROMPT.format(n=floor_n)},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
            ],
        }],
    }).encode()

    req = urllib.request.Request(
        config.OLMX_HOST + "/chat/completions",
        body,
        {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.OLMX_API_KEY}",
        },
    )

    last = ""
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=config.LLM_TIMEOUT) as resp:
                r = json.load(resp)
            txt = r["choices"][0]["message"]["content"].strip()
            data = _extract_json(txt)
            if data is None:
                last = f"no JSON in reply: {txt[:80]!r}"
                continue
            total = data.get("total")
            if total is None:
                last = f"no 'total' field: {txt[:80]!r}"
                continue
            return int(total)
        except Exception as e:  # noqa: BLE001 - fail soft, report last error
            last = repr(e)[:150]
    raise LLMError(last)
