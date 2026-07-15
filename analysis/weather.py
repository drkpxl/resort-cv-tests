"""Weather from a single frame using brightness / contrast / color stats.

This camera sees almost no sky, so we read the weather off the scene:
- night  -> the whole frame is dark
- snow   -> a big fraction of the ground is bright and colorless
- sunny  -> hard shadows (high contrast) and vivid color (high saturation)
- rainy  -> what's left: flat gray light, washed-out color (includes overcast)
"""

import cv2
import numpy as np

import config
from analysis.motion import roi_slice


def analyze(frame: np.ndarray) -> dict:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_luma = float(gray.mean())
    contrast = float(gray.std())

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mean_sat = float(hsv[:, :, 1].mean())

    ground = cv2.cvtColor(roi_slice(frame, config.GROUND_ROI), cv2.COLOR_BGR2HSV)
    white = (ground[:, :, 2] > config.SNOW_VALUE_MIN) & (
        ground[:, :, 1] < config.SNOW_SAT_MAX
    )
    white_fraction = float(np.count_nonzero(white)) / white.size

    if mean_luma < config.NIGHT_MEAN_LUMA:
        label = "night"
    elif white_fraction > config.SNOW_WHITE_FRACTION:
        label = "snow"
    elif contrast > config.SUNNY_CONTRAST_MIN and mean_sat > config.SUNNY_SAT_MIN:
        label = "sunny"
    else:
        label = "rainy"

    return {
        "weather": label,
        "brightness": round(mean_luma, 1),
        "contrast": round(contrast, 1),
        "saturation": round(mean_sat, 1),
        "white_fraction": round(white_fraction, 3),
    }


if __name__ == "__main__":
    frame = cv2.imread("frame_a.jpg")
    print(analyze(frame))
