"""Model bake-off: run several detectors over the same frame and compare.

Usage:
    uv run python -m analysis.compare              # grab a live frame
    uv run python -m analysis.compare shot.jpg     # use a saved frame

For each (model, input size) combo it saves an annotated image to
compare_out/ and prints a table: people found, mean confidence, inference
time. Ground truth is your eyeballs - open the images and count who was
missed. The interesting comparisons for this camera:

- yolov8n (CNN, 3M params) vs rtdetr-l (transformer, 32M params): does
  deformable attention actually handle the overlapping-people-in-a-queue
  case better, as the DETR family claims?
- 640 vs 1280 input: how much of the tiny-people problem is architecture
  vs simply resolution?
"""

import sys
import time
from pathlib import Path

import cv2

import config
from analysis import grab

RUNS = [
    ("yolov8n", 640),
    ("yolov8n", 1280),
    ("rtdetr-l", 640),
    ("rtdetr-l", 1280),
]

_loaded: dict[str, object] = {}


def load_model(name: str):
    if name not in _loaded:
        from ultralytics import RTDETR, YOLO
        cls = RTDETR if name.startswith("rtdetr") else YOLO
        _loaded[name] = cls(f"{name}.pt")
    return _loaded[name]


def run_one(name: str, imgsz: int, frame, out_dir: Path) -> dict:
    model = load_model(name)
    # warm-up pass so timing measures inference, not first-call setup
    model.predict(frame, conf=config.PERSON_CONFIDENCE, classes=[0],
                  imgsz=imgsz, verbose=False)
    start = time.perf_counter()
    results = model.predict(frame, conf=config.PERSON_CONFIDENCE, classes=[0],
                            imgsz=imgsz, verbose=False)
    elapsed_ms = (time.perf_counter() - start) * 1000

    boxes = results[0].boxes
    confs = [float(c) for c in boxes.conf]

    annotated = frame.copy()
    for xyxy, conf in zip(boxes.xyxy, confs):
        x1, y1, x2, y2 = map(int, xyxy.tolist())
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"{conf:.2f}", (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    label = f"{name} @ {imgsz}: {len(confs)} people, {elapsed_ms:.0f}ms"
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(annotated, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)
    out_path = out_dir / f"{name}_{imgsz}.jpg"
    cv2.imwrite(str(out_path), annotated)

    return {
        "model": name,
        "imgsz": imgsz,
        "count": len(confs),
        "mean_conf": sum(confs) / len(confs) if confs else 0.0,
        "ms": elapsed_ms,
        "image": str(out_path),
    }


def main():
    if len(sys.argv) > 1:
        frame = cv2.imread(sys.argv[1])
        if frame is None:
            sys.exit(f"could not read {sys.argv[1]}")
        source = sys.argv[1]
    else:
        frame, _ = grab.grab_pair(gap_seconds=0.5)
        source = "live grab"

    out_dir = Path("compare_out")
    out_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(out_dir / "input.jpg"), frame)

    print(f"source: {source}  ({frame.shape[1]}x{frame.shape[0]})\n")
    print(f"{'model':<10} {'imgsz':>5} {'people':>6} {'mean conf':>9} {'ms':>7}")
    for name, imgsz in RUNS:
        r = run_one(name, imgsz, frame, out_dir)
        print(f"{r['model']:<10} {r['imgsz']:>5} {r['count']:>6} "
              f"{r['mean_conf']:>9.2f} {r['ms']:>7.0f}")
    print(f"\nannotated frames in {out_dir}/ - open them and count the misses.")


if __name__ == "__main__":
    main()
