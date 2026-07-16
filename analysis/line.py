"""Lift-line length via YOLOv8n person detection.

The model finds every person in the frame; we then keep only the ones
whose feet (bottom-center of their bounding box) land inside the queue
polygon, so mini-golfers and hikers elsewhere don't count as a lift line.
"""

import cv2
import numpy as np

import config

_model = None


def _get_model():
    global _model
    if _model is None:
        from ultralytics import RTDETR, YOLO
        cls = RTDETR if config.DETECTOR_MODEL.startswith("rtdetr") else YOLO
        _model = cls(f"{config.DETECTOR_MODEL}.pt")
    return _model


def _polygon_pixels(frame_shape) -> np.ndarray:
    h, w = frame_shape[:2]
    return np.array(
        [(int(x * w), int(y * h)) for x, y in config.QUEUE_POLYGON], dtype=np.int32
    )


def analyze(frame: np.ndarray) -> dict:
    model = _get_model()
    results = model.predict(frame, conf=config.PERSON_CONFIDENCE, classes=[0],
                            imgsz=config.YOLO_IMAGE_SIZE, verbose=False)

    polygon = _polygon_pixels(frame.shape)
    in_queue, elsewhere = [], []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        feet = (int((x1 + x2) / 2), int(y2))
        entry = {
            "box": (int(x1), int(y1), int(x2), int(y2)),
            "conf": round(float(box.conf[0]), 2),
        }
        if cv2.pointPolygonTest(polygon, feet, False) >= 0:
            in_queue.append(entry)
        else:
            elsewhere.append(entry)

    count = len(in_queue)
    if count >= config.LINE_LONG_MIN:
        status = "long"
    elif count >= config.LINE_SHORT_MIN:
        status = "short"
    else:
        status = "none"

    return {
        "person_count": count,
        "line_status": status,
        "people_in_queue": in_queue,
        "people_elsewhere": elsewhere,
    }


if __name__ == "__main__":
    frame = cv2.imread("frame_a.jpg")
    result = analyze(frame)
    print(
        f"queue={result['person_count']} ({result['line_status']}), "
        f"elsewhere={len(result['people_elsewhere'])}"
    )
