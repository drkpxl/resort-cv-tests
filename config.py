"""Central config: stream source, regions of interest, thresholds, cadence.

ROIs are expressed as fractions of frame width/height (0.0-1.0) so they
survive resolution changes. Tune them against the debug view at /debug.
"""

STREAM_URL = "https://www.youtube.com/watch?v=vyWAzJAWv4w"
YOUTUBE_VIDEO_ID = "vyWAzJAWv4w"

# Frame grabbing
GRAB_MAX_HEIGHT = 720          # yt-dlp format cap
STREAM_URL_TTL_SECONDS = 4 * 3600  # re-resolve HLS URL after this age
ANALYZE_INTERVAL_SECONDS = 15  # full analysis cadence
MOTION_FRAME_GAP_SECONDS = 2.0  # gap between the two frames diffed for motion

# --- Gondola motion (classical CV: frame differencing) ---
# Rectangle over the cable span, upper-right of frame: (x1, y1, x2, y2)
MOTION_ROI = (0.60, 0.05, 0.92, 0.40)
MOTION_PIXEL_THRESHOLD = 25    # 0-255 gray delta considered "changed"
MOTION_SCORE_THRESHOLD = 0.005  # fraction of ROI pixels changed => moving

# --- Lift line (YOLOv8n person detection) ---
# Polygon over the queue plaza, lower-left of frame: [(x, y), ...]
QUEUE_POLYGON = [
    (0.02, 0.62),
    (0.46, 0.62),
    (0.46, 0.80),
    (0.02, 0.80),
]
PERSON_CONFIDENCE = 0.25
YOLO_IMAGE_SIZE = 1280         # match native stream width; default 640 misses
                               # the tiny (~20px) people in this wide shot
# Which detector the app uses: "yolov8n" (CNN, 3M params, fast) or
# "rtdetr-l" (transformer, 32M params, claims better occlusion handling -
# relevant because a lift line IS overlapping people). Weights auto-download.
# Compare them on real frames with: uv run python -m analysis.compare
DETECTOR_MODEL = "yolov8n"
LINE_SHORT_MIN = 4             # people count => "short" at this many
LINE_LONG_MIN = 11             # people count => "long" at this many

# --- Weather (classical CV: brightness / contrast / color stats) ---
# Camera location, for computing sun elevation. Dusk light is statistically
# identical to overcast (flat, gray, desaturated) and auto-exposure hides
# darkness, so pixel stats alone can't tell sunset from rain. Astronomy can.
RESORT_LAT = 39.8868
RESORT_LON = -105.7625
NIGHT_SUN_ELEVATION = -6.0     # sun below this (civil twilight ends) => night
TWILIGHT_SUN_ELEVATION = 6.0   # sun below this => twilight; the resort sits
                               # in a valley, so light goes flat well before
                               # geometric sunset
# Note: this camera sees almost no sky, so we infer weather from light on
# the scene itself: shadows (contrast), color saturation, ground whiteness.
NIGHT_MEAN_LUMA = 45.0         # mean gray below this => night
SNOW_WHITE_FRACTION = 0.30     # frac of ground pixels bright+desaturated => snow
SNOW_VALUE_MIN = 190           # HSV V (0-255) for "white"
SNOW_SAT_MAX = 40              # HSV S (0-255) for "white"
SUNNY_CONTRAST_MIN = 32.0      # luma std-dev; hard shadows => high contrast
SUNNY_SAT_MIN = 50.0           # mean saturation; sunlight => vivid color
# Ground region sampled for snow coverage (avoids buildings/trees at top)
GROUND_ROI = (0.05, 0.55, 0.95, 0.98)

# --- Real-time tracking mode (/static/live.html) ---
# Continuous decode + YOLO tracking. 4 fps leaves headroom: one YOLO pass
# at imgsz 1280 takes ~100-200ms on the M4, and falling behind the live
# stream makes the ffmpeg pipe back up and lag grow unboundedly.
REALTIME_FPS = 4
REALTIME_WIDTH = 1280
REALTIME_HEIGHT = 720
TRAIL_LENGTH = 40              # tracked positions kept per person (~10s)

# --- Parking-lot capacity (separate experiment; different stream) ---
# High-oblique event-lot cam. Same COCO detector as the lift line, but on
# vehicle classes, plus segmentation for true silhouette coverage.
PARKING_STREAM_URL = "https://www.youtube.com/watch?v=ta-4QZbfMXc"
# Segmentation model gives per-vehicle silhouette masks (not just boxes), so
# coverage measures painted metal, not the gaps between cars. Drop-in sibling
# of yolov8n; weights auto-download on first run.
PARKING_SEG_MODEL = "yolov8n-seg"
PARKING_CONFIDENCE = 0.25
PARKING_IMAGE_SIZE = 1280       # wide shot; far cars are ~15px (gondola lesson)
# COCO vehicle class ids: 2 car, 3 motorcycle, 5 bus, 7 truck
PARKING_VEHICLE_CLASSES = [2, 3, 5, 7]
PARKING_CAPACITY = 300          # denominator for the count-based estimate;
                                # a starting guess, resort is confirming real #
# Lot ground-plane quad as fractional coords, ordered
# far-left(vanishing) -> far-right -> near-right -> near-left. Tune in the
# debug image. This defines both what counts as "in the lot" and the
# homography used to flatten perspective.
PARKING_LOT_QUAD = [
    (0.14, 0.28),
    (0.98, 0.42),
    (0.98, 0.98),
    (0.05, 0.95),
]
# Bird's-eye output size (top-down). Assumes the lot runs deeper (toward the
# vanishing point) than it is wide; purely a rendering/weighting choice.
PARKING_BEV_WIDTH = 600
PARKING_BEV_HEIGHT = 900
# Fullness bucket edges as occupied-fraction cutoffs -> labels below in code:
# <5% empty, then quarter / half / three-quarter / full at the midpoints.
PARKING_BUCKET_EDGES = (0.05, 0.375, 0.625, 0.875)

# --- Storage / server ---
DB_PATH = "data.db"
DEBUG_FRAME_PATH = "static/debug_frame.jpg"
HISTORY_HOURS = 2              # how much history /api/status returns
STALE_AFTER_SECONDS = 90       # latest observation older than this => stale
