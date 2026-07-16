"""Weather from a single frame using brightness / contrast / color stats,
gated by sun position.

This camera sees almost no sky, so we read the weather off the scene:
- night    -> the sun is well below the horizon (or the frame is dark)
- twilight -> the sun is too low to judge: dusk light is statistically
              identical to overcast, so we refuse to guess sunny/rainy
- snow     -> a big fraction of the ground is bright and colorless
- sunny    -> hard shadows (high contrast) and vivid color (high saturation)
- rainy    -> what's left: flat gray light, washed-out color (incl. overcast)

The sun-elevation gate is sensor fusion in miniature: pixels can't separate
sunset from rain, but latitude + longitude + a clock can.
"""

import math
from datetime import datetime, timezone

import cv2
import numpy as np

import config
from analysis.motion import roi_slice


def solar_elevation(dt_utc: datetime | None = None) -> float:
    """Sun elevation in degrees at the resort (NOAA approximation, ~0.5deg)."""
    if dt_utc is None:
        dt_utc = datetime.now(timezone.utc)
    days = (dt_utc - datetime(2000, 1, 1, 12, tzinfo=timezone.utc)).total_seconds() / 86400.0
    mean_long = math.radians((280.460 + 0.9856474 * days) % 360)
    mean_anom = math.radians((357.528 + 0.9856003 * days) % 360)
    ecliptic_long = mean_long + math.radians(
        1.915 * math.sin(mean_anom) + 0.020 * math.sin(2 * mean_anom)
    )
    obliquity = math.radians(23.439 - 0.0000004 * days)
    right_asc = math.atan2(
        math.cos(obliquity) * math.sin(ecliptic_long), math.cos(ecliptic_long)
    )
    declination = math.asin(math.sin(obliquity) * math.sin(ecliptic_long))
    sidereal_hours = (18.697374558 + 24.06570982441908 * days) % 24
    hour_angle = math.radians((sidereal_hours * 15 + config.RESORT_LON) % 360) - right_asc
    lat = math.radians(config.RESORT_LAT)
    return math.degrees(
        math.asin(
            math.sin(lat) * math.sin(declination)
            + math.cos(lat) * math.cos(declination) * math.cos(hour_angle)
        )
    )


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

    sun_elev = solar_elevation()
    if sun_elev < config.NIGHT_SUN_ELEVATION or mean_luma < config.NIGHT_MEAN_LUMA:
        label = "night"
    elif sun_elev < config.TWILIGHT_SUN_ELEVATION:
        label = "twilight"
    elif white_fraction > config.SNOW_WHITE_FRACTION:
        label = "snow"
    elif contrast > config.SUNNY_CONTRAST_MIN and mean_sat > config.SUNNY_SAT_MIN:
        label = "sunny"
    else:
        label = "rainy"

    return {
        "weather": label,
        "sun_elevation": round(sun_elev, 1),
        "brightness": round(mean_luma, 1),
        "contrast": round(contrast, 1),
        "saturation": round(mean_sat, 1),
        "white_fraction": round(white_fraction, 3),
    }


if __name__ == "__main__":
    frame = cv2.imread("frame_a.jpg")
    print(analyze(frame))
