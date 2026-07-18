"""Vision floor: YOLO segmentation + SAHI tiling.

Returns a high-recall floor count of vehicles inside the lot polygon, a
perspective-corrected surface-coverage %, and an overlay image (red vehicle
silhouettes + green footprint dots, no box/text) for the LLM add-misses step.
Ported from the CV-Test `analysis/parking.py` prototype.
"""
import cv2
import numpy as np

import config

_model = None


def _get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(f"{config.VISION_MODEL}.pt")
    return _model


def _quad_pixels(shape) -> np.ndarray:
    h, w = shape[:2]
    return np.array([(x * w, y * h) for x, y in config.LOT_POLYGON], dtype=np.float32)


def _homography(quad: np.ndarray) -> np.ndarray:
    # Map the lot quad to a unit rectangle. The destination size is arbitrary:
    # it scales the coverage weight map uniformly, which cancels in the coverage
    # ratio. The near/far weighting comes entirely from the quad's placement.
    dst = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    return cv2.getPerspectiveTransform(quad, dst)


def _weight_map(hom: np.ndarray, shape) -> np.ndarray:
    """Per-pixel real-ground-area weight = |det(H)| / |w|^3 (homography Jacobian)."""
    h, w = shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    denom = hom[2, 0] * xs + hom[2, 1] * ys + hom[2, 2]
    denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
    return abs(np.linalg.det(hom)) / np.abs(denom) ** 3


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def _detect_tiled(frame: np.ndarray):
    """SAHI-style sliced detection; returns (union vehicle mask, kept dets)."""
    model = _get_model()
    H, W = frame.shape[:2]
    rows, cols = config.TILES
    th, tw = int(H / rows), int(W / cols)
    ov_h, ov_w = int(th * config.OVERLAP), int(tw * config.OVERLAP)

    mask = np.zeros((H, W), dtype=np.uint8)
    raw = []
    for r in range(rows):
        for c in range(cols):
            y0, x0 = max(0, r * th - ov_h), max(0, c * tw - ov_w)
            y1 = min(H, (r + 1) * th + ov_h)
            x1 = min(W, (c + 1) * tw + ov_w)
            tile = frame[y0:y1, x0:x1]
            res = model.predict(tile, conf=config.CONF,
                                classes=config.VEHICLE_CLASSES,
                                imgsz=config.YOLO_IMGSZ, verbose=False)[0]
            polys = res.masks.xy if res.masks is not None else []
            for box, poly in zip(res.boxes, polys):
                bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                g = (bx1 + x0, by1 + y0, bx2 + x0, by2 + y0)
                cv2.fillPoly(mask, [(poly + [x0, y0]).astype(np.int32)], 1)
                raw.append((g, (int((g[0] + g[2]) / 2), int(g[3])), float(box.conf[0])))

    raw.sort(key=lambda d: d[2], reverse=True)
    kept = []
    for det in raw:
        if all(_iou(det[0], k[0]) < 0.45 for k in kept):
            kept.append(det)
    return mask, kept


def detect(frame: np.ndarray):
    """Return (vision_count, coverage_pct, overlay_bgr)."""
    quad = _quad_pixels(frame.shape)
    quad_i = quad.astype(np.int32)
    lot = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(lot, [quad_i], 1)
    lot_bool = lot.astype(bool)

    mask, dets = _detect_tiled(frame)
    foots_in, foots_out = [], []
    for _box, foot, _conf in dets:
        (foots_in if cv2.pointPolygonTest(quad_i, foot, False) >= 0
         else foots_out).append(foot)

    vehicle_in_lot = mask.astype(bool) & lot_bool
    hom = _homography(quad)
    weight = _weight_map(hom, frame.shape)
    lot_weight = weight[lot_bool].sum() or 1.0
    coverage_pct = float(weight[vehicle_in_lot].sum() / lot_weight * 100)

    # Overlay: red silhouettes + green footprint dots, no lot box, no text.
    tint = frame.copy()
    tint[mask.astype(bool)] = (0, 0, 255)
    overlay = cv2.addWeighted(tint, 0.4, frame, 0.6, 0)
    for fx, fy in foots_in + foots_out:
        cv2.circle(overlay, (fx, fy), 4, (0, 255, 0), -1)

    return len(foots_in), coverage_pct, overlay
