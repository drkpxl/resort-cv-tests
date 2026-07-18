"""All tunables for the parking monitor, in one place."""

# --- Source ---------------------------------------------------------------
YOUTUBE_URL = "https://www.youtube.com/watch?v=4a-3iEM7bHk"  # Brighton lot
GRAB_MAX_HEIGHT = 720

# --- Lot geometry ---------------------------------------------------------
# Full frame by default — count every vehicle the detector finds, so the
# boundary never clips real cars. Only draw a tighter polygon (fractional
# corners far-left -> far-right -> near-right -> near-left) if this camera sees
# vehicles OUTSIDE the lot (a through-road, a neighboring lot) that inflate the
# count. Check any custom polygon by drawing it on a frame first.
LOT_POLYGON = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
CAPACITY = 150  # rough guess; only used for the (uncalibrated) count/capacity %

# --- Vision floor (YOLO detector) -----------------------------------------
VISION_MODEL = "yolov8x-seg"   # high recall; weights auto-download (~140MB)
YOLO_IMGSZ = 1280
CONF = 0.15                     # lower = more recall on distant/dusk cars
VEHICLE_CLASSES = [2, 3, 5, 7]  # COCO: car, motorcycle, bus, truck
TILES = (5, 6)                  # SAHI grid (rows, cols); (6,8)+ over-fragments here
OVERLAP = 0.25

# --- LLM add-misses (local oMLX, OpenAI-compatible) -------------------------
OLMX_HOST = "http://127.0.0.1:8000/v1"
OLMX_API_KEY = "sk-olmx"
LLM_MODEL = "unsloth-gemma-4-E4B-it-qat-oQ4"
LLM_THINK = False               # thinking spirals on small models — keep OFF
LLM_TIMEOUT = 180

# --- Schedule / storage / web ---------------------------------------------
INTERVAL_MINUTES = 15
DATA_DIR = "data"
IMAGE_RETENTION_DAYS = 14       # prune image/thumb files older than this (0 = keep all)
WEB_PORT = 8600
WEB_ROW_LIMIT = 200
