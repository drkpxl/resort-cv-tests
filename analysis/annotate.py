"""Draw the CV work onto a frame so you can see what the analyses see."""

import cv2
import numpy as np

import config
from analysis.line import _polygon_pixels


def annotate(frame: np.ndarray, motion: dict, line: dict, weather: dict) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    # Motion ROI (cyan) with per-pixel change mask overlaid in red
    x1, y1, x2, y2 = config.MOTION_ROI
    mx1, my1, mx2, my2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
    mask = motion.get("changed_mask")
    if mask is not None:
        region = out[my1:my2, mx1:mx2]
        mask = mask[: region.shape[0], : region.shape[1]]
        region[mask] = (0, 0, 255)
    cv2.rectangle(out, (mx1, my1), (mx2, my2), (255, 255, 0), 2)
    cv2.putText(
        out,
        f"motion {motion['motion_score']:.4f} {'MOVING' if motion['moving'] else 'still'}",
        (mx1 + 4, my1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2,
    )

    # Queue polygon (yellow) and person boxes: green in queue, gray elsewhere
    cv2.polylines(out, [_polygon_pixels(out.shape)], True, (0, 255, 255), 2)
    for person in line["people_in_queue"]:
        bx1, by1, bx2, by2 = person["box"]
        cv2.rectangle(out, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
    for person in line["people_elsewhere"]:
        bx1, by1, bx2, by2 = person["box"]
        cv2.rectangle(out, (bx1, by1), (bx2, by2), (160, 160, 160), 1)

    # Stats banner along the bottom
    banner = (
        f"queue: {line['person_count']} ({line['line_status']})   "
        f"weather: {weather['weather']}  "
        f"luma {weather['brightness']} contrast {weather['contrast']} "
        f"sat {weather['saturation']} white {weather['white_fraction']}"
    )
    cv2.rectangle(out, (0, h - 28), (w, h), (0, 0, 0), -1)
    cv2.putText(out, banner, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)
    return out
