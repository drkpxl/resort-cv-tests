"""Parking-lot fullness: empty / 25 / 50 / 75 / full, three ways.

The camera looks down the lot at a steep oblique angle, so it has a strong
vanishing point: a car parked near the camera fills ~30x the pixels of an
identical car parked at the back. That single fact is what makes this
interesting, because it breaks the two obvious approaches in opposite ways:

  1. COUNT     - detect vehicles, count the ones whose footprint lands in the
                 lot, divide by an assumed capacity. Under-counts badly at the
                 back, where distant cars merge into a blob the detector misses.

  2. COVERAGE  - what fraction of the lot's surface is covered by vehicle
     (naive)    silhouettes (segmentation masks, not boxes). Over-weights the
                 near foreground: one close truck swings it more than a whole
                 back row.

  3. COVERAGE  - the same silhouette coverage, but every pixel is weighted by
     (corrected) how much real ground it represents. A homography maps the lot
                 quad to a flat top-down rectangle; the local area-scale of that
                 map is its Jacobian determinant, which for a homography has the
                 closed form det(H)/w(x,y)^3. So the perspective weight of each
                 pixel is simply 1 / w^3 - no hand-tuned distance curve. Far
                 pixels (small w, near the horizon) weigh the most, exactly
                 undoing the foreground bias.

We show all three side by side, plus a bird's-eye warp of the lot, so the
disagreements are visible. Run:  uv run python -m analysis.parking
"""

import base64
import json
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

import config

_BUCKET_LABELS = [
    "Empty",
    "~25% (Quarter)",
    "~50% (Half)",
    "~75% (Three-quarter)",
    "Full (at capacity)",
]

_model = None


def _get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO

        _model = YOLO(f"{config.PARKING_SEG_MODEL}.pt")
    return _model


def _quad_pixels(frame_shape) -> np.ndarray:
    """Lot quad as an int32 pixel polygon (far-L, far-R, near-R, near-L)."""
    h, w = frame_shape[:2]
    return np.array(
        [(x * w, y * h) for x, y in config.PARKING_LOT_QUAD], dtype=np.float32
    )


def _bucket(frac: float) -> str:
    """Map an occupied fraction (0-1) to one of the five fullness labels."""
    for edge, label in zip(config.PARKING_BUCKET_EDGES, _BUCKET_LABELS):
        if frac < edge:
            return label
    return _BUCKET_LABELS[-1]


def _homography(quad: np.ndarray):
    """Homography from the image lot-quad to a flat top-down rectangle."""
    W, H = config.PARKING_BEV_WIDTH, config.PARKING_BEV_HEIGHT
    dst = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float32)
    return cv2.getPerspectiveTransform(quad, dst), (W, H)


def _perspective_weight_map(homography: np.ndarray, shape) -> np.ndarray:
    """Per-pixel real-ground-area weight = |det(H)| / |w(x,y)|^3.

    For a homography, the Jacobian determinant of the point map is
    det(H) / w^3 where w = h20*x + h21*y + h22 is the homogeneous denominator.
    That determinant is precisely how much top-down (uniform) ground area one
    image pixel spans, i.e. the perspective weight we want. det(H) is constant,
    so only the 1/w^3 term matters for the ratios we compute.
    """
    h, w = shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    denom = homography[2, 0] * xs + homography[2, 1] * ys + homography[2, 2]
    # Guard the horizon (w -> 0), where weight blows up to infinity.
    denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
    weight = abs(np.linalg.det(homography)) / np.abs(denom) ** 3
    return weight


def _detect(frame: np.ndarray):
    """Single whole-frame pass. Returns (vehicle_mask, [(box, foot)])."""
    model = _get_model()
    results = model.predict(
        frame,
        conf=config.PARKING_CONFIDENCE,
        classes=config.PARKING_VEHICLE_CLASSES,
        imgsz=config.PARKING_IMAGE_SIZE,
        verbose=False,
    )[0]
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    dets = []
    polys = results.masks.xy if results.masks is not None else []
    for box, poly in zip(results.boxes, polys):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
        dets.append(((x1, y1, x2, y2), (int((x1 + x2) / 2), int(y2)),
                     float(box.conf[0])))
    return mask, dets


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def _detect_tiled(frame: np.ndarray, rows: int, cols: int, overlap: float):
    """Sliced (SAHI-style) detection: run the model on overlapping crops so a
    far car that is ~15px in the full frame becomes ~60px in its tile, then
    merge across tiles with greedy NMS so cars on a seam aren't double-counted.
    """
    model = _get_model()
    H, W = frame.shape[:2]
    th, tw = int(H / rows), int(W / cols)
    ov_h, ov_w = int(th * overlap), int(tw * overlap)

    mask = np.zeros((H, W), dtype=np.uint8)
    raw = []  # (box_xyxy_global, foot, conf)
    for r in range(rows):
        for c in range(cols):
            y0 = max(0, r * th - ov_h)
            x0 = max(0, c * tw - ov_w)
            y1 = min(H, (r + 1) * th + ov_h)
            x1 = min(W, (c + 1) * tw + ov_w)
            tile = frame[y0:y1, x0:x1]
            res = model.predict(
                tile, conf=config.PARKING_CONFIDENCE,
                classes=config.PARKING_VEHICLE_CLASSES,
                imgsz=config.PARKING_IMAGE_SIZE, verbose=False,
            )[0]
            polys = res.masks.xy if res.masks is not None else []
            for box, poly in zip(res.boxes, polys):
                bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                g = (bx1 + x0, by1 + y0, bx2 + x0, by2 + y0)
                cv2.fillPoly(mask, [(poly + [x0, y0]).astype(np.int32)], 1)
                raw.append((g, (int((g[0] + g[2]) / 2), int(g[3])),
                            float(box.conf[0])))

    # Greedy NMS on the merged boxes (coverage mask is a union, so it needs no
    # dedup — overlapping paint is idempotent — but the count does).
    raw.sort(key=lambda d: d[2], reverse=True)
    kept = []
    for det in raw:
        if all(_iou(det[0], k[0]) < 0.45 for k in kept):
            kept.append(det)
    return mask, kept


def _summarize(frame, vehicle_mask, dets) -> dict:
    """Turn a vehicle mask + detections into the three fullness readings."""
    quad = _quad_pixels(frame.shape)
    quad_i = quad.astype(np.int32)
    lot_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(lot_mask, [quad_i], 1)
    lot_bool = lot_mask.astype(bool)

    footprints_in_lot, footprints_out = [], []
    for _box, foot, _conf in dets:
        if cv2.pointPolygonTest(quad_i, foot, False) >= 0:
            footprints_in_lot.append(foot)
        else:
            footprints_out.append(foot)
    vehicle_in_lot = vehicle_mask.astype(bool) & lot_bool

    # 1. Count-based.
    count = len(footprints_in_lot)
    count_frac = min(count / config.PARKING_CAPACITY, 1.0)

    # 2. Coverage, naive (raw pixels).
    lot_px = int(lot_bool.sum()) or 1
    cover_naive = vehicle_in_lot.sum() / lot_px

    # 3. Coverage, perspective-corrected (Jacobian-weighted pixels).
    homography, bev_size = _homography(quad)
    weight = _perspective_weight_map(homography, frame.shape)
    lot_weight = weight[lot_bool].sum() or 1.0
    cover_corrected = weight[vehicle_in_lot].sum() / lot_weight

    return {
        "count": count,
        "count_frac": count_frac,
        "count_bucket": _bucket(count_frac),
        "cover_naive": cover_naive,
        "cover_naive_bucket": _bucket(cover_naive),
        "cover_corrected": cover_corrected,
        "cover_corrected_bucket": _bucket(cover_corrected),
        "footprints_in_lot": footprints_in_lot,
        "footprints_out": footprints_out,
        "vehicle_mask": vehicle_mask,
        "lot_quad": quad_i,
        "homography": homography,
        "bev_size": bev_size,
    }


def analyze(frame: np.ndarray) -> dict:
    mask, dets = _detect(frame)
    return _summarize(frame, mask, dets)


def analyze_tiled(frame: np.ndarray, rows: int = 3, cols: int = 4,
                  overlap: float = 0.2) -> dict:
    mask, dets = _detect_tiled(frame, rows, cols, overlap)
    return _summarize(frame, mask, dets)


def render(frame: np.ndarray, result: dict, title: str = "") -> np.ndarray:
    """Side-by-side: annotated original | bird's-eye warp of the lot."""
    left = frame.copy()

    # Tint vehicle silhouettes red inside the lot.
    veh = result["vehicle_mask"].astype(bool)
    tint = left.copy()
    tint[veh] = (0, 0, 255)
    left = cv2.addWeighted(tint, 0.35, left, 0.65, 0)

    # Lot outline + footprints (green = counted in lot, gray = outside).
    cv2.polylines(left, [result["lot_quad"]], True, (0, 255, 0), 2)
    for fx, fy in result["footprints_in_lot"]:
        cv2.circle(left, (fx, fy), 4, (0, 255, 0), -1)
    for fx, fy in result["footprints_out"]:
        cv2.circle(left, (fx, fy), 4, (150, 150, 150), -1)

    lines = []
    if title:
        lines.append(title)
    lines += [
        f"COUNT       {result['count']}/{config.PARKING_CAPACITY}"
        f"  ({result['count_frac'] * 100:4.0f}%)  {result['count_bucket']}",
        f"COVER naive         {result['cover_naive'] * 100:4.0f}%"
        f"  {result['cover_naive_bucket']}",
        f"COVER corrected     {result['cover_corrected'] * 100:4.0f}%"
        f"  {result['cover_corrected_bucket']}",
    ]
    for i, text in enumerate(lines):
        y = 30 + i * 30
        cv2.putText(left, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(left, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)

    return left


def render_hybrid(frame: np.ndarray, result: dict) -> np.ndarray:
    """Detections-only image for the 'hybrid' LLM test: red vehicle silhouettes
    and a green dot at every footprint — but NO lot box and NO stats. The LLM
    sees what the detector found and decides fullness/extent for itself.
    """
    veh = result["vehicle_mask"].astype(bool)
    tint = frame.copy()
    tint[veh] = (0, 0, 255)
    out = cv2.addWeighted(tint, 0.4, frame, 0.6, 0)
    for fx, fy in result["footprints_in_lot"] + result["footprints_out"]:
        cv2.circle(out, (fx, fy), 4, (0, 255, 0), -1)
    return out


_VLM_SCHEMA = {
    "type": "object",
    "properties": {
        "vehicle_count": {"type": "integer"},
        "fullness": {
            "type": "string",
            "enum": ["Empty", "~25%", "~50%", "~75%", "Full"],
        },
        "reasoning": {"type": "string"},
    },
    "required": ["vehicle_count", "fullness", "reasoning"],
    "additionalProperties": False,
}

_VLM_PROMPT = (
    "This is a single frame from a fixed camera looking down a parking lot at an "
    "oblique angle (near cars large, far cars tiny and overlapping). Estimate:\n"
    "1. vehicle_count: roughly how many vehicles (cars, trucks, RVs, vans) are "
    "parked in the lot. A best-effort number is fine.\n"
    "2. fullness: how full the lot is, as one of Empty / ~25% / ~50% / ~75% / Full.\n"
    "3. reasoning: one sentence on how you judged it.\n"
    "Ignore tents, people, and bikes — count only vehicles."
)


def vlm_estimate(frame: np.ndarray, model: str = "claude-haiku-4-5") -> dict:
    """Ask a vision LLM the same question, as a baseline vs. the CV pipeline.

    Needs ANTHROPIC_API_KEY in the environment. Sends one JPEG frame; the model
    never sees the lot quad, so it judges fullness from the raw image alone.
    """
    import anthropic

    ok, buf = cv2.imencode(".jpg", frame)
    b64 = base64.standard_b64encode(buf.tobytes()).decode()

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        output_config={"format": {"type": "json_schema", "schema": _VLM_SCHEMA}},
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": _VLM_PROMPT},
            ],
        }],
    )
    text = next(b.text for b in resp.content if b.type == "text")
    out = json.loads(text)
    out["model"] = model
    out["input_tokens"] = resp.usage.input_tokens
    out["output_tokens"] = resp.usage.output_tokens
    return out


def _grab_live(url: str) -> np.ndarray:
    """Resolve the YouTube URL and pull one frame (standalone; no cache)."""
    hls = subprocess.run(
        ["yt-dlp", "-g", "-f", f"best[height<={config.GRAB_MAX_HEIGHT}]", url],
        capture_output=True, text=True, timeout=60,
    ).stdout.strip().splitlines()[0]
    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-i", hls, "-frames:v", "1",
             "-q:v", "2", "-y", tmp.name],
            capture_output=True, timeout=45,
        )
        return cv2.imread(tmp.name)


if __name__ == "__main__":
    # Optional image-path arg lets you re-run on a saved frame; else grab live.
    if len(sys.argv) > 1:
        frame = cv2.imread(sys.argv[1])
    else:
        print("grabbing live parking frame...")
        frame = _grab_live(config.PARKING_STREAM_URL)

    result = analyze(frame)
    out = "parking_debug.jpg"
    cv2.imwrite(out, render(frame, result))
    print(
        f"CV pipeline:\n"
        f"  count           = {result['count']}/{config.PARKING_CAPACITY} "
        f"({result['count_frac'] * 100:.0f}%)  -> {result['count_bucket']}\n"
        f"  cover naive     = {result['cover_naive'] * 100:.0f}%  "
        f"-> {result['cover_naive_bucket']}\n"
        f"  cover corrected = {result['cover_corrected'] * 100:.0f}%  "
        f"-> {result['cover_corrected_bucket']}"
    )

    # VLM baseline: only if a key is available (skips cleanly otherwise).
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            v = vlm_estimate(frame)
            print(
                f"VLM ({v['model']}):\n"
                f"  count ~ {v['vehicle_count']}   fullness -> {v['fullness']}\n"
                f"  \"{v['reasoning']}\"\n"
                f"  ({v['input_tokens']} in / {v['output_tokens']} out tokens)"
            )
        except Exception as e:
            print(f"VLM step failed: {e}")
    else:
        print("VLM: skipped (set ANTHROPIC_API_KEY to run the Haiku baseline)")

    print(f"wrote {out}")
