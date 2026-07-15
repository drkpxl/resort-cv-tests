"""Gondola motion via frame differencing — the 'hello world' of video CV.

Two frames a couple seconds apart are converted to grayscale and
subtracted. Pixels whose brightness changed more than a threshold count
as 'motion'. We only look inside a rectangle drawn over the cable span,
so swaying trees and mini-golfers elsewhere don't fool us.
"""

import cv2
import numpy as np

import config


def roi_slice(frame: np.ndarray, roi: tuple[float, float, float, float]):
    """Crop a frame to a fractional (x1, y1, x2, y2) rectangle."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = roi
    return frame[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]


def analyze(frame_a: np.ndarray, frame_b: np.ndarray) -> dict:
    gray_a = cv2.cvtColor(roi_slice(frame_a, config.MOTION_ROI), cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(roi_slice(frame_b, config.MOTION_ROI), cv2.COLOR_BGR2GRAY)

    # Blur before diffing so sensor noise and compression artifacts cancel out
    gray_a = cv2.GaussianBlur(gray_a, (5, 5), 0)
    gray_b = cv2.GaussianBlur(gray_b, (5, 5), 0)

    diff = cv2.absdiff(gray_a, gray_b)
    changed = diff > config.MOTION_PIXEL_THRESHOLD
    score = float(np.count_nonzero(changed)) / changed.size

    return {
        "moving": score > config.MOTION_SCORE_THRESHOLD,
        "motion_score": round(score, 5),
        "changed_mask": changed,  # for the debug overlay
    }


if __name__ == "__main__":
    a = cv2.imread("frame_a.jpg")
    b = cv2.imread("frame_b.jpg")
    result = analyze(a, b)
    print(f"moving={result['moving']} score={result['motion_score']}")
