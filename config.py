LINE_POSITION = 290
OFFSET = 20   # widened from 10 — a narrow zone lets fast vehicles jump clean over it between frames

FRAME_SKIP = 1   # keep at 1 for reliable tracking/counting — see project notes
PIXEL_TO_METER = 0.05
FPS = 30

# ── Detection/working resolution ──────────────────────────────────────────────
DETECT_W, DETECT_H = 640, 480
DETECT_CONF      = 0.25   # inference confidence passed to model.track()
COUNT_CONFIDENCE = 0.30   # currently unused by the proximity-trigger counting logic (kept for optional reintroduction)

# ── Classification heuristic tuning ───────────────────────────────────────────
BUS_MIN_AREA_FRAC      = 0.030
TRUCK_MIN_AREA_FRAC    = 0.050
TRICYCLE_MIN_AREA_FRAC = 0.020
MIN_AREA_FOR_RECLASSIFICATION = 0.008   # below this box size, trust the raw model label — geometry is unreliable at this scale
HIGH_CONF_TRUST     = 0.70   # above this confidence, trust the model outright, skip the heuristic
LABEL_SWITCH_MARGIN = 1.3    # a challenger label must beat the incumbent by this multiplier to switch
LABEL_HISTORY_SIZE  = 10

# ── Counting zone (derived) ───────────────────────────────────────────────────
LINE_UPPER = LINE_POSITION - OFFSET
LINE_LOWER = LINE_POSITION + OFFSET

# ── File paths ─────────────────────────────────────────────────────────────────
# Centralized here so they're not hardcoded across app.py/main.py separately.
MODEL_PATH = "best 4.pt"
VIDEO_PATH = "folder/328225909-busy-hadejia-road-traffic-and-.mp4"   # used by main.py; app.py takes an uploaded path instead — edit THIS if you change test videos, not a local copy in main.py

# ── Speed ──────────────────────────────────────────────────────────────────────
SPEED_LIMIT_KMH = 60   # used for an "OVERSPEEDING" flag if you enable it later

# ── Congestion thresholds (vehicle count in frame) ────────────────────────────
# Numeric thresholds only — colors are kept per-file since app.py needs BGR
# (OpenCV) and plot.py needs hex, which don't share one representation cleanly.
CONGESTION_THRESHOLDS = [
    (5,   "FREE FLOW"),
    (15,  "MODERATE"),
    (999, "CONGESTED"),
]

# ── Level of Service thresholds (HCM standard, based on traffic density) ─────
LOS_THRESHOLDS = [
    (11,  "A", "Free Flow"),
    (18,  "B", "Reasonable Free Flow"),
    (26,  "C", "Stable Flow"),
    (35,  "D", "Approaching Unstable"),
    (45,  "E", "Unstable Flow"),
    (999, "F", "Forced/Breakdown"),
]
