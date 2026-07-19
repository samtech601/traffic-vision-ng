import math
from collections import deque
from config import PIXEL_TO_METER, FRAME_SKIP, FPS

# ─── SPEED HISTORY PER VEHICLE ────────────────────────────────────────────────
# Stores last N speed readings per track_id for smoothing
_speed_history = {}
HISTORY_SIZE   = 8   # number of readings to average over


def estimate_speed(track, track_id=None):
    """
    Estimate vehicle speed from its position history.

    How it works:
    1. Calculate speed between multiple recent point pairs
    2. Remove outliers (very high or very low readings)
    3. Average remaining readings for a stable result
    4. Apply exponential smoothing using history per vehicle

    Args:
        track    : list of (x, y) center positions of the vehicle
        track_id : unique ID of the vehicle (for per-vehicle smoothing)

    Returns:
        Smoothed speed in km/h (float), or 0 if not enough data
    """

    # Need at least 3 points to calculate anything meaningful
    if len(track) < 3:
        return 0

    # ── Step 1: Calculate multiple speed samples ───────────────────────────
    raw_speeds = []
    num_pairs  = min(6, len(track) - 1)  # use up to 6 recent pairs

    for i in range(1, num_pairs + 1):
        x1, y1 = track[-i - 1]
        x2, y2 = track[-i]

        pixel_distance = math.hypot(x2 - x1, y2 - y1)

        # Skip if vehicle barely moved (stationary or noise)
        if pixel_distance < 1.0:
            continue

        meter_distance = pixel_distance * PIXEL_TO_METER
        time_taken     = FRAME_SKIP / FPS

        if time_taken <= 0:
            continue

        speed_kmh = (meter_distance / time_taken) * 3.6
        raw_speeds.append(speed_kmh)

    if not raw_speeds:
        return 0

    # ── Step 2: Remove outliers ────────────────────────────────────────────
    # Anything above 200 km/h or below 1 km/h is likely noise
    filtered = [s for s in raw_speeds if 1.0 <= s <= 200.0]

    if not filtered:
        return 0

    # Further filter: remove readings far from the median
    if len(filtered) >= 3:
        median    = sorted(filtered)[len(filtered) // 2]
        threshold = median * 0.6   # allow 60% deviation from median
        filtered  = [s for s in filtered
                     if abs(s - median) <= threshold]

    if not filtered:
        return 0

    # ── Step 3: Average the clean readings ────────────────────────────────
    current_speed = sum(filtered) / len(filtered)

    # ── Step 4: Exponential smoothing using vehicle history ───────────────
    # This prevents sudden jumps between frames
    if track_id is not None:
        if track_id not in _speed_history:
            _speed_history[track_id] = deque(maxlen=HISTORY_SIZE)

        history = _speed_history[track_id]
        history.append(current_speed)

        if len(history) >= 2:
            # Weight recent readings more heavily
            weights       = [0.5 ** (len(history) - 1 - i)
                             for i in range(len(history))]
            total_weight  = sum(weights)
            smoothed      = sum(w * s for w, s in zip(weights, history))
            current_speed = smoothed / total_weight

    return round(current_speed, 1)


def clear_speed_history(track_id):
    """
    Remove speed history for a vehicle that has left the frame.
    Call this when a track ID disappears to free memory.
    """
    if track_id in _speed_history:
        del _speed_history[track_id]


def get_speed_category(speed_kmh):
    """
    Categorise speed into traffic flow labels.
    Useful for display and reporting.

    Returns:
        String label: 'stationary', 'slow', 'normal', 'fast', 'speeding'
    """
    if speed_kmh < 5:
        return "stationary"
    elif speed_kmh < 20:
        return "slow"
    elif speed_kmh < 60:
        return "normal"
    elif speed_kmh < 100:
        return "fast"
    else:
        return "speeding"
