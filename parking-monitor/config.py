"""All tunables for the parking monitor, in one place."""

# --- Source ---------------------------------------------------------------
YOUTUBE_URL = "https://www.youtube.com/watch?v=4a-3iEM7bHk"  # Brighton lot
GRAB_MAX_HEIGHT = 720

# --- Lot geometry ---------------------------------------------------------
# Fractional (0-1) polygon over the parking surface, ordered
# far-left -> far-right -> near-right -> near-left. Tune per camera.
LOT_POLYGON = [(0.02, 0.44), (0.85, 0.31), (0.99, 0.99), (0.01, 0.99)]
CAPACITY = 150  # rough guess; only used for the (uncalibrated) count/capacity %

# --- Vision floor (YOLO detector) -----------------------------------------
VISION_MODEL = "yolov8x-seg"   # high recall; weights auto-download (~140MB)
YOLO_IMGSZ = 1280
CONF = 0.25
VEHICLE_CLASSES = [2, 3, 5, 7]  # COCO: car, motorcycle, bus, truck
TILES = (4, 5)                  # SAHI grid (rows, cols); raise for more recall
OVERLAP = 0.25
BEV_SIZE = (600, 900)           # top-down rect for the perspective-weight homography

# --- LLM add-misses (local Ollama) ----------------------------------------
OLLAMA_HOST = "http://localhost:11434"
LLM_MODEL = "gemma4:e4b"
LLM_THINK = False               # thinking spirals on small models — keep OFF
LLM_KEEP_ALIVE = "30s"          # unload between cycles (tight-RAM box)
LLM_TIMEOUT = 180

# --- Schedule / storage / web ---------------------------------------------
INTERVAL_MINUTES = 15
DATA_DIR = "data"
IMAGE_RETENTION_DAYS = 14       # prune image/thumb files older than this (0 = keep all)
WEB_PORT = 8600
WEB_ROW_LIMIT = 200
