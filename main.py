import cv2
import csv
import time
import os
import sys
from collections import defaultdict, deque
from datetime import datetime
from ultralytics import YOLO

from config import *
from speed import estimate_speed
from plot import generate_graphs

# ─── UNIVERSITY / PROJECT INFO ────────────────────────────────────────────────
UNIVERSITY = "YOUR UNIVERSITY NAME"   # ← Change to your actual university

# ─── VEHICLE CLASS IDs ────────────────────────────────────────────────────────
# ⚠️ IMPORTANT: These IDs must match model.names EXACTLY for your CURRENT
# trained model — confirmed correct for this model as of this version.
# If you ever swap in a different .pt file, re-run this check first:
#   from ultralytics import YOLO
#   print(YOLO("best1.pt").names)
#vehicle_classes = {
# #   0: "car",
 #   1: "tricycle",
 #   2: "bus",
 #   3: "truck",
 #   4: "motorbike",
 #   5: "van",
#}
vehicle_classes={0: 'bus', 1: 'car', 2: 'motorbike', 3: 'threewheel', 4: 'truck', 5: 'van'}

# ─── COLOR PER VEHICLE TYPE (BGR) ─────────────────────────────────────────────
# A cohesive "neon dashboard" palette — each color is vivid and visually
# distinct from the others, and distinct from the red/amber used by the
# zone and UI overlays elsewhere, so nothing gets confused for a status color.
BOX_COLORS = {
    "car":        (255, 191,   0),   # vivid sky blue
    "truck":      ( 28, 159, 255),   # amber orange
    "bus":        (113, 204,  46),   # emerald green
    "motorbike":  (153,   0, 255),   # hot pink / magenta
    "tricycle":   (  0, 214, 255),   # golden yellow
    "van":        (182,  89, 155),   # soft lavender purple
}
# ─── CONGESTION / LOS DISPLAY COLORS (BGR) ────────────────────────────────────
# ✅ FIXED: thresholds themselves come from config.py (CONGESTION_THRESHOLDS,
# LOS_THRESHOLDS) — this is only the color each label maps to on screen.
# Previously this file had its own hardcoded CONGESTION_LEVELS/LOS_LEVELS with
# threshold numbers baked in, completely separate from config.py — editing
# config.py had NO effect on this file. That's fixed now.
CONGESTION_COLORS = {
    "FREE FLOW": (0, 255,   0),
    "MODERATE":  (0, 255, 255),
    "CONGESTED": (0,   0, 255),
}
LOS_COLORS = {
    "A": (0, 255,   0), "B": (0, 255, 128), "C": (0, 255, 255),
    "D": (0, 165, 255), "E": (0,  80, 255), "F": (0,   0, 255),
}

# ─── SETTINGS ─────────────────────────────────────────────────────────────────
# ✅ #5 — DETECT_CONF, COUNT_CONFIDENCE, DETECT_W/H, VIDEO_PATH, the
# classification heuristic thresholds, LINE_UPPER/LINE_LOWER, MODEL_PATH,
# CONGESTION_THRESHOLDS, and LOS_THRESHOLDS all come from config.py via
# `from config import *` above — genuinely single-sourced now, not just
# duplicated locally with the same starting values.
MAX_TRACK_HISTORY   = 50     # max position history per vehicle
INFER_EVERY_N       = FRAME_SKIP  # run detection every Nth frame (perf), still render/write every frame


# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────

def get_congestion(vehicle_count):
    for threshold, label in CONGESTION_THRESHOLDS:
        if vehicle_count <= threshold:
            return label, CONGESTION_COLORS.get(label, (255, 255, 255))
    return "CONGESTED", CONGESTION_COLORS["CONGESTED"]


def get_los(density):
    for threshold, letter, desc in LOS_THRESHOLDS:
        if density <= threshold:
            return letter, desc, LOS_COLORS.get(letter, (255, 255, 255))
    return "F", "Forced/Breakdown", LOS_COLORS["F"]


def get_flow_rate(vehicle_timestamps, window_seconds=60):
    current_time = time.time()
    cutoff_time  = current_time - window_seconds
    recent       = [t for t in vehicle_timestamps if t > cutoff_time]
    flow_rate    = len(recent) * (3600 / window_seconds)
    return round(flow_rate, 1), len(recent)


def get_movement_direction(position_history, window=7, min_net_displacement=3):
    """
    ✅ Net displacement over the last `window` position samples, not a
    majority vote of frame-to-frame deltas. In slow or stop-and-go traffic,
    per-frame movement is often just 1-2px of noise with roughly equal ups
    and downs — a majority vote ties out to "ambiguous" constantly, even
    when the vehicle has clearly, steadily drifted across the line over the
    whole window. Comparing the first and last sample in the window instead
    smooths past that jitter and reflects real net movement.

    Returns "DOWN", "UP", or None if there's not enough history or the net
    movement is smaller than min_net_displacement (genuinely stationary/
    too little data to tell — NOT evidence of the wrong direction).
    """
    recent = position_history[-window:]
    if len(recent) < 2:
        return None
    net = recent[-1][1] - recent[0][1]
    if net > min_net_displacement:
        return "DOWN"
    elif net < -min_net_displacement:
        return "UP"
    return None


def get_peak_minute(traffic_log):
    if not traffic_log:
        return "N/A"
    minute_counts = defaultdict(int)
    for entry in traffic_log:
        minute = entry["time"].strftime("%H:%M")
        minute_counts[minute] += 1
    peak = max(minute_counts, key=minute_counts.get)
    return f"{peak} ({minute_counts[peak]} vehicles)"


def print_summary(counts, traffic_log, avg_speeds,
                  peak_flow, avg_density, los_letter):
    grand_total = sum(v["IN"] + v["OUT"] for v in counts.values())
    print("\n" + "="*60)
    print("         TRAFFIC ANALYSIS SUMMARY REPORT")
    print("="*60)
    print(f"  Total Vehicles : {grand_total}")
    for v_label, v_counts in counts.items():
        total = v_counts["IN"] + v_counts["OUT"]
        pct   = (total / grand_total * 100) if grand_total > 0 else 0
        print(f"  {v_label:<12} IN={v_counts['IN']:>3}  "
              f"OUT={v_counts['OUT']:>3}  TOTAL={total:>3}  ({pct:.1f}%)")
    print("-"*60)
    if avg_speeds:
        print(f"  Avg Speed    : {sum(avg_speeds)/len(avg_speeds):.1f} km/h")
    print(f"  Peak Flow    : {peak_flow} veh/hr")
    print(f"  Avg Density  : {avg_density:.1f} veh/km")
    print(f"  LOS          : {los_letter}")
    peak = get_peak_minute(traffic_log)
    print(f"  Peak Minute  : {peak}")
    print("="*60 + "\n")


def finish_system():
    cap.release()
    out.release()
    csv_file.close()
    cv2.destroyAllWindows()
    flow_rate, _ = get_flow_rate(vehicle_timestamps)
    avg_spd      = (sum(speed_log) / len(speed_log)) if speed_log else 0
    density      = (flow_rate / avg_spd) if avg_spd > 0 else 0
    los_letter, _, _ = get_los(density)
    print_summary(counts, traffic_log, speed_log,
                  flow_rate, density, los_letter)
    print("Generating graphs...")
    try:
        generate_graphs()
        print("Graphs saved to graphs/ folder")
    except Exception as e:
        print(f"Graph error: {e}")
    print("Done!")


# ─── LOAD MODEL ───────────────────────────────────────────────────────────────
# ✅ #4 — clear, actionable error messages instead of a raw crash/traceback.
if not os.path.exists(MODEL_PATH):
    print(f"\n❌ Model file not found: '{MODEL_PATH}'")
    print(f"   Place your trained model in this folder, or update MODEL_PATH in config.py.")
    sys.exit(1)

try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"\n❌ Could not load model '{MODEL_PATH}': {e}")
    sys.exit(1)

print("Model classes:", model.names)  # sanity check — confirm this matches vehicle_classes above

# ─── LOAD VIDEO ───────────────────────────────────────────────────────────────
if not os.path.exists(VIDEO_PATH):
    print(f"\n❌ Video file not found: '{VIDEO_PATH}'")
    sys.exit(1)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"\n❌ Could not open video '{VIDEO_PATH}' — it may be corrupted or in an unsupported format.")
    sys.exit(1)

orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

FPS    = cap.get(cv2.CAP_PROP_FPS) or 30

# ─── VIDEO WRITER ─────────────────────────────────────────────────────────────
# ✅ FIXED: We now write EVERY frame (not just processed ones), so output FPS
# matches the source FPS and playback speed stays correct.
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
out    = cv2.VideoWriter("DEMO_method7.avi", fourcc, FPS, (orig_w, orig_h))

# ─── DATA STORAGE ─────────────────────────────────────────────────────────────
track_history         = defaultdict(list)
counted_ids           = set()  # ✅ one set — a track is counted at most once, period, regardless of direction
counts                = {v: {"OUT": 0, "IN": 0}
                         for v in set(vehicle_classes.values())}
traffic_log           = []
speed_log             = []
vehicle_label_history = defaultdict(
    lambda: deque(maxlen=LABEL_HISTORY_SIZE))
confirmed_labels      = {}   # track_id -> currently displayed label (hysteresis)
recently_counted_pos  = {}   # legacy, kept only in case you want it back — unused now
last_seen_frame        = {}  # ✅ track_id -> frame_id it was last detected in, for stale cleanup
speed_history          = defaultdict(lambda: deque(maxlen=5))  # ✅ per-track recent speeds, for smoothing
STALE_TRACK_FRAMES     = 300  # remove a track's state after this many frames unseen (well beyond track_buffer=60, so we never wipe a legitimately-buffered track early)
vehicle_timestamps    = []
flow_history          = deque(maxlen=10)
density_history       = deque(maxlen=10)
peak_flow_rate        = 0

# ─── CSV OUTPUT ───────────────────────────────────────────────────────────────
csv_file   = open("output.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "Time", "Vehicle", "Direction", "Speed(km/h)",
    "Confidence", "Congestion", "Flow_Rate", "Density", "LOS"
])

# ─── MAIN LOOP ────────────────────────────────────────────────────────────────
frame_id            = 0
paused              = False
last_metrics_update = time.time()
flow_rate           = 0
density             = 0
los_letter          = "A"
los_desc            = "Free Flow"
los_color           = (0, 255, 0)
current_frame_vehicles = 0
frame               = None   # holds the current display frame across iterations

# ✅ #14 — FPS tracking (actual processing throughput, not source video FPS)
fps_frame_counter = 0
fps_timer_start   = time.time()
display_fps       = 0.0

print("Controls: P=Pause  S=Screenshot  R=Reset  Q/ESC=Quit")

while True:

    if not paused:
        ret, new_frame = cap.read()
        if not ret:
            print("\nVideo finished.")
            finish_system()
            break

        frame_id += 1
        # Resize to the SAME dimensions/method used when the training images
        # were exported from Roboflow (stretch-to-fit, not letterboxed).
        # This is the working frame for detection, tracking, AND all drawing —
        # we only convert back to native resolution when writing the output video.
        frame = cv2.resize(new_frame, (DETECT_W, DETECT_H))
        h, w  = DETECT_H, DETECT_W

        run_detection = (frame_id % INFER_EVERY_N == 0)

        if run_detection:
            results = model.track(
                frame, persist=True,
                conf=DETECT_CONF,
                iou=0.45,
                tracker="custom_bytetrack.yaml",  # tuned thresholds — see custom_bytetrack.yaml
                verbose=False
            )

            # ✅ #14 — FPS: measured over a rolling 1-second window.
            fps_frame_counter += 1
            current_time = time.time()
            if current_time - fps_timer_start >= 1.0:
                display_fps = round(fps_frame_counter / (current_time - fps_timer_start), 1)
                fps_frame_counter = 0
                fps_timer_start = current_time

            # ── Update traffic metrics every second ──────────────────────────
            if current_time - last_metrics_update >= 1.0:
                flow_rate, _ = get_flow_rate(vehicle_timestamps)
                recent_speeds = list(speed_log)[-50:]
                avg_spd = (sum(recent_speeds) / len(recent_speeds)) if recent_speeds else 1
                density = flow_rate / max(avg_spd, 1)
                flow_history.append(flow_rate)
                density_history.append(density)
                if flow_rate > peak_flow_rate:
                    peak_flow_rate = flow_rate
                los_letter, los_desc, los_color = get_los(density)
                last_metrics_update = current_time

            current_frame_vehicles = 0

            if results[0].boxes.id is not None:
                boxes   = results[0].boxes.xyxy.cpu().numpy()
                ids     = results[0].boxes.id.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                confs   = results[0].boxes.conf.cpu().numpy()

                current_frame_vehicles = sum(
                    1 for c in classes if int(c) in vehicle_classes
                )

                frame_area = w * h

                for box, track_id, cls, conf in zip(boxes, ids, classes, confs):
                    cls      = int(cls)
                    track_id = int(track_id)

                    if cls not in vehicle_classes:
                        continue

                    label = vehicle_classes[cls]

                    x1, y1, x2, y2 = map(int, box)
                    box_width  = x2 - x1
                    box_height = y2 - y1
                    box_area   = box_width * box_height
                    area_frac  = box_area / frame_area if frame_area > 0 else 0

                    # Skip oversized false detections
                    if box_width > w * 0.7 or box_height > h * 0.7:
                        continue

                    aspect_ratio = box_width / max(box_height, 1)

                    # ✅ FIXED: only apply heuristic reclassification when the
                    # model itself is UNSURE. If the model is already confident
                    # in its own prediction, trust it — don't let an aspect-ratio
                    # guess override a correct high-confidence detection.
                    if conf < HIGH_CONF_TRUST and area_frac >= MIN_AREA_FOR_RECLASSIFICATION:
                        if label == "car":
                            if area_frac > TRUCK_MIN_AREA_FRAC and aspect_ratio < 1.5:
                                label = "truck"
                            elif area_frac > BUS_MIN_AREA_FRAC and aspect_ratio > 1.6:
                                label = "bus"

                        if label == "motorbike" and area_frac > TRICYCLE_MIN_AREA_FRAC and aspect_ratio < 1.8:
                            label = "tricycle"

                    # ✅ FIXED: confidence-weighted voting instead of a flat
                    # majority count. A handful of noisy low-confidence frames
                    # can no longer outvote a smaller number of high-confidence
                    # ones, and a hysteresis margin stops the displayed label
                    # from flickering between two close contenders frame-to-frame.
                    vehicle_label_history[track_id].append((label, conf))
                    history = vehicle_label_history[track_id]

                    weighted_scores = defaultdict(float)
                    for hist_label, hist_conf in history:
                        weighted_scores[hist_label] += hist_conf

                    ranked = sorted(weighted_scores.items(), key=lambda kv: kv[1], reverse=True)
                    top_label, top_score = ranked[0]

                    current_confirmed = confirmed_labels.get(track_id)
                    if current_confirmed is None:
                        confirmed_labels[track_id] = top_label
                    elif top_label != current_confirmed:
                        current_score = weighted_scores.get(current_confirmed, 0.0)
                        # Require the challenger to clearly beat the incumbent
                        # before switching, not just narrowly edge it out.
                        if top_score > current_score * LABEL_SWITCH_MARGIN:
                            confirmed_labels[track_id] = top_label

                    best_label = confirmed_labels[track_id]
                    color      = BOX_COLORS.get(best_label, (255, 255, 255))

                    # ✅ Bottom-center anchor instead of centroid. The centroid
                    # shifts around based on box height (a tall truck's center
                    # sits well above where it actually touches the road), so
                    # it's a noisier reference for line-crossing than the
                    # bottom edge, which tracks the vehicle's road contact
                    # point much more consistently.
                    cx = (x1 + x2) // 2
                    cy = y2

                    track_history[track_id].append((cx, cy))
                    if len(track_history[track_id]) > MAX_TRACK_HISTORY:
                        track_history[track_id].pop(0)

                    for px, py in track_history[track_id][-8:]:
                        cv2.circle(frame, (px, py), 2, color, -1)

                    last_seen_frame[track_id] = frame_id

                    speed = estimate_speed(track_history[track_id], track_id)
                    if speed > 0:
                        speed_log.append(speed)
                        speed_history[track_id].append(speed)
                    # ✅ Smooth the displayed/logged speed with a rolling
                    # average instead of one noisy instantaneous estimate.
                    if speed_history[track_id]:
                        smoothed_speed = round(
                            sum(speed_history[track_id]) / len(speed_history[track_id]), 1)
                    else:
                        smoothed_speed = speed
                    speed = smoothed_speed

                    # ── Counting logic ────────────────────────────────────────
                    # ✅ PROXIMITY TRIGGER — adapted from a much simpler design
                    # that turned out to work more reliably in practice: count
                    # a vehicle the FIRST time its position lands inside the
                    # zone, deduped by track_id so it can never count twice.
                    # No persistent "which side was it on before" state to get
                    # stuck in — that entire class of bug (tracks first seen
                    # already inside/past the zone, low-confidence frames
                    # eating the only counting attempt, etc.) simply can't
                    # happen anymore, because there's no state to lose.
                    #
                    # Direction (IN/OUT) is worked out AFTER the fact, from
                    # this track's own trajectory so far, not from a
                    # persistent flag — comparing where it entered the frame
                    # to where it is now.
                    if LINE_UPPER <= cy <= LINE_LOWER and track_id not in counted_ids:
                        counted_ids.add(track_id)

                        history_pts = track_history[track_id]
                        if len(history_pts) >= 2:
                            net_movement = history_pts[-1][1] - history_pts[0][1]
                            direction = "IN" if net_movement >= 0 else "OUT"
                        else:
                            direction = "IN"  # not enough history yet — default

                        counts[best_label][direction] += 1
                        now_time = time.time()
                        vehicle_timestamps.append(now_time)
                        now = datetime.now()
                        congestion_label, _ = get_congestion(current_frame_vehicles)
                        traffic_log.append({"time": now, "vehicle": best_label})
                        csv_writer.writerow([
                            now.strftime("%Y-%m-%d %H:%M:%S"),
                            best_label, direction, speed,
                            f"{conf:.2f}", congestion_label,
                            flow_rate, f"{density:.2f}", los_letter
                        ])


                    # ── Draw bounding box and label ───────────────────────────
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    txt       = f"{best_label} {conf:.0%} | ID:{track_id} | {speed}km/h"
                    text_size = cv2.getTextSize(
                        txt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
                    cv2.rectangle(frame,
                                  (x1, y1 - 18),
                                  (x1 + text_size[0] + 4, y1),
                                  color, -1)
                    cv2.putText(frame, txt, (x1 + 2, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
                    cv2.circle(frame, (cx, cy), 4, color, -1)

            # ✅ Periodic stale-track cleanup. Runs every 60 frames rather than
            # every single frame (cheap enough, no need to do it constantly).
            # A track is only removed once it's been unseen for far longer
            # than the tracker's own track_buffer (60 frames in
            # custom_bytetrack.yaml) — so this only clears out tracks that
            # have genuinely left the scene, never ones mid-occlusion that
            # ByteTrack itself is still trying to re-identify.
            if frame_id % 60 == 0:
                stale_ids = [
                    tid for tid, last_seen in last_seen_frame.items()
                    if frame_id - last_seen > STALE_TRACK_FRAMES
                ]
                for tid in stale_ids:
                    track_history.pop(tid, None)
                    vehicle_label_history.pop(tid, None)
                    confirmed_labels.pop(tid, None)
                    speed_history.pop(tid, None)
                    last_seen_frame.pop(tid, None)
                    # Deliberately NOT removing from counted_ids —
                    # it must stay permanent for the life of the program so
                    # a track_id can never be double-counted if it somehow
                    # reappears later.

        # ── UI overlays: drawn on EVERY frame, using the latest known state ──
        # (so output video stays smooth/full-length even on frames where
        # detection itself was skipped for performance)

        # ✅ Redesigned zone visual: no solid center "counting line" anymore —
        # just a soft translucent band marking the trigger zone, with clear
        # directional arrows showing which way counts as IN vs OUT. Green =
        # IN (moving down/toward camera), amber-red = OUT (moving away) —
        # a common, intuitive convention that reads at a glance.
        IN_COLOR  = (100, 220,  80)   # soft green
        OUT_COLOR = ( 60,  90, 255)   # soft red-orange

        zone_overlay = frame.copy()
        cv2.rectangle(zone_overlay, (0, LINE_UPPER), (w, LINE_LOWER),
                      (235, 235, 235), -1)
        cv2.addWeighted(zone_overlay, 0.10, frame, 0.90, 0, frame)

        cv2.line(frame, (0, LINE_UPPER), (w, LINE_UPPER), (235, 235, 235), 1)
        cv2.line(frame, (0, LINE_LOWER), (w, LINE_LOWER), (235, 235, 235), 1)

        zone_mid_y = (LINE_UPPER + LINE_LOWER) // 2

        # IN arrow (downward) — left side
        arrow_x = 50
        cv2.arrowedLine(frame, (arrow_x, zone_mid_y - 22), (arrow_x, zone_mid_y + 22),
                        IN_COLOR, 3, tipLength=0.4)
        cv2.putText(frame, "IN", (arrow_x + 12, zone_mid_y + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, IN_COLOR, 2)

        # OUT arrow (upward) — right side
        arrow_x2 = w - 90
        cv2.arrowedLine(frame, (arrow_x2, zone_mid_y + 22), (arrow_x2, zone_mid_y - 22),
                        OUT_COLOR, 3, tipLength=0.4)
        cv2.putText(frame, "OUT", (arrow_x2 + 12, zone_mid_y + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, OUT_COLOR, 2)

        congestion_label, congestion_color = get_congestion(current_frame_vehicles)
        cv2.rectangle(frame, (w - 173, 3), (w - 5, 32), (0, 0, 0), -1)
        cv2.putText(frame, f"TRAFFIC:{congestion_label}",
                    (w - 170, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, congestion_color, 2)

        panel_x = 10
        panel_y = 22
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5),
                      (360, 22 + len(counts) * 22), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        font = cv2.FONT_HERSHEY_SIMPLEX
        for v_label, v_counts in counts.items():
            panel_color = BOX_COLORS.get(v_label, (255, 255, 255))
            total       = v_counts["IN"] + v_counts["OUT"]

            x = panel_x
            name_txt = f"{v_label:<10}"
            cv2.putText(frame, name_txt, (x, panel_y), font, 0.46, panel_color, 1)
            x += cv2.getTextSize(name_txt, font, 0.46, 1)[0][0]

            in_txt = f"IN={v_counts['IN']:>3}  "
            cv2.putText(frame, in_txt, (x, panel_y), font, 0.46, IN_COLOR, 1)
            x += cv2.getTextSize(in_txt, font, 0.46, 1)[0][0]

            out_txt = f"OUT={v_counts['OUT']:>3}  "
            cv2.putText(frame, out_txt, (x, panel_y), font, 0.46, OUT_COLOR, 1)
            x += cv2.getTextSize(out_txt, font, 0.46, 1)[0][0]

            cv2.putText(frame, f"TOTAL={total:>3}", (x, panel_y), font, 0.46, (255, 255, 255), 1)
            panel_y += 22

        grand_total = sum(v["IN"] + v["OUT"] for v in counts.values())
        cv2.rectangle(frame, (0, h - 35), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, f"TOTAL: {grand_total}",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 255), 2)

        if speed_log:
            recent_speeds = speed_log[-50:]
            avg_spd = sum(recent_speeds) / len(recent_speeds)
            cv2.putText(frame, f"AVG SPEED: {avg_spd:.1f} km/h",
                        (w - 220, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 2)

        metrics_overlay = frame.copy()
        cv2.rectangle(metrics_overlay,
                      (0, h - 89), (220, h - 35), (0, 0, 0), -1)
        cv2.addWeighted(metrics_overlay, 0.55, frame, 0.45, 0, frame)

        # ✅ #14/#21 — FPS + vehicles currently on road
        cv2.putText(frame, f"FPS: {display_fps}  |  ON ROAD: {current_frame_vehicles}",
                    (10, h - 82), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (200, 200, 200), 1)
        cv2.putText(frame, f"FLOW: {flow_rate:.0f} veh/hr",
                    (10, h - 68), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (200, 200, 200), 1)
        cv2.putText(frame, f"DENSITY: {density:.1f} veh/km",
                    (10, h - 54), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (200, 200, 200), 1)
        cv2.putText(frame, f"LOS: {los_letter} - {los_desc}",
                    (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, los_color, 1)

        cv2.putText(frame, UNIVERSITY,
                    (w // 2 - len(UNIVERSITY) * 4, h - 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    # ── Paused overlay ────────────────────────────────────────────────────────
    if paused and frame is not None:
        h, w = frame.shape[:2]
        cv2.putText(frame, "PAUSED",
                    (w // 2 - 50, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

    if frame is not None:
        cv2.imshow("Nigerian Traffic Detection System", frame)
        # Write every frame (fixes the old skipped-frame speed-up bug), scaled
        # back up to native resolution for the saved output video only —
        # detection/drawing still happens at DETECT_W x DETECT_H above.
        output_frame = cv2.resize(frame, (orig_w, orig_h))
        out.write(output_frame)

    # ── Keyboard controls ─────────────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF

    if key == 27 or key == ord("q"):
        print("User quit.")
        finish_system()
        break

    elif key == ord("p"):
        paused = not paused
        print("PAUSED" if paused else "RESUMED")

    elif key == ord("s") and frame is not None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"screenshot_{ts}.jpg", frame)
        print(f"Screenshot saved → screenshot_{ts}.jpg")

    elif key == ord("r"):
        for v in counts:
            counts[v] = {"IN": 0, "OUT": 0}
        counted_ids.clear()
        traffic_log.clear()
        speed_log.clear()
        vehicle_label_history.clear()
        confirmed_labels.clear()
        recently_counted_pos.clear()
        last_seen_frame.clear()
        speed_history.clear()
        vehicle_timestamps.clear()
        flow_history.clear()
        density_history.clear()
        peak_flow_rate = 0
        print("Counts reset!")

# ─── CLEANUP ──────────────────────────────────────────────────────────────────
try:
    cap.release()
    out.release()
    csv_file.close()
    cv2.destroyAllWindows()
except Exception:
    pass
