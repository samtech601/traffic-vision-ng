import os
import cv2
import csv
import time
import shutil
import threading
from collections import defaultdict, deque
from datetime import datetime
from flask import (Flask, render_template, request,
                   redirect, url_for, Response,
                   jsonify, send_file)
from ultralytics import YOLO

from speed import estimate_speed
from plot import generate_graphs
from config import (
    LINE_POSITION, OFFSET, FRAME_SKIP, PIXEL_TO_METER,
    DETECT_W, DETECT_H, LINE_UPPER, LINE_LOWER,
    BUS_MIN_AREA_FRAC, TRUCK_MIN_AREA_FRAC, TRICYCLE_MIN_AREA_FRAC,
    MIN_AREA_FOR_RECLASSIFICATION, HIGH_CONF_TRUST, LABEL_SWITCH_MARGIN,
    LABEL_HISTORY_SIZE, MODEL_PATH, CONGESTION_THRESHOLDS, LOS_THRESHOLDS,
)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"   # kept for reference; use UPLOAD_DIR (absolute) below instead
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

# ✅ Absolute paths anchored to THIS file's folder, not the process's current
# working directory. Relative paths like "output.csv" break depending on how
# the app is launched (double-click, VS Code debugger, terminal in a
# different folder, Render's working directory, etc.) — this fixes that
# category of bug entirely, which is what caused the WinError 2 you hit.
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR     = os.path.join(BASE_DIR, "uploads")
CSV_PATH       = os.path.join(BASE_DIR, "output.csv")
GRAPHS_DIR     = os.path.join(BASE_DIR, "graphs")
STATIC_GRAPHS_DIR = os.path.join(BASE_DIR, "static", "graphs")

os.makedirs(UPLOAD_DIR,         exist_ok=True)
os.makedirs(STATIC_GRAPHS_DIR,  exist_ok=True)
os.makedirs(GRAPHS_DIR,         exist_ok=True)

# ── Vehicle classes ───────────────────────────────────────────────────────────
# ⚠️ Must match your trained model.names EXACTLY. Verify with:
#   from ultralytics import YOLO; print(YOLO("best1.pt").names)
VEHICLE_CLASSES = {
    0: "car",
    1: "tricycle",
    2: "bus",
    3: "truck",
    4: "motorbike",
    5: "van",
}

# ── Color palette (BGR) — cohesive "neon dashboard" look, matches the ────────
# standalone detection script and the dashboard UI's color-coded rows.
BOX_COLORS = {
    "car":        (255, 191,   0),   # vivid sky blue
    "truck":      ( 28, 159, 255),   # amber orange
    "bus":        (113, 204,  46),   # emerald green
    "motorbike":  (153,   0, 255),   # hot pink / magenta
    "tricycle":   (  0, 214, 255),   # golden yellow
    "van":        (182,  89, 155),   # soft lavender purple
}
IN_COLOR_BGR  = (100, 220,  80)   # soft green
OUT_COLOR_BGR = ( 60,  90, 255)   # soft red-orange

CONGESTION_LEVELS = [
    (5,   "FREE FLOW", (34, 197,  94)),
    (15,  "MODERATE",  (234, 179,  8)),
    (999, "CONGESTED", (239,  68, 68)),
]

# ── Global shared state ───────────────────────────────────────────────────────
state = {
    "running":     False,
    "finished":    False,
    "use_camera":  False,
    "counts":      {v: {"IN": 0, "OUT": 0} for v in VEHICLE_CLASSES.values()},
    "speed_log":   [],
    "traffic_log": [],
    "congestion":  "FREE FLOW",
    "current_on_road": 0,     # ✅ #21 — vehicles visible in the current frame
    "flow_rate":   0.0,       # ✅ #20 — vehicles/hour, computed from a rolling window
    "density":     0.0,       # ✅ #20 — vehicles/km, derived from flow ÷ avg speed
    "los_letter":  "A",       # ✅ #20 — HCM Level of Service (A best, F worst)
    "los_desc":    "Free Flow",
    "peak_minute": "N/A",     # ✅ #22 — busiest minute so far this run
    "fps":         0.0,       # ✅ #14 — actual processing throughput
    "frame":       None,
    "error":       None,
}
state_lock  = threading.Lock()

# ── Camera frame generator (separate thread) ──────────────────────────────────
camera_frame = None
camera_lock  = threading.Lock()
camera_thread_running = False


def get_congestion(count):
    for threshold, label, _ in CONGESTION_LEVELS:
        if count <= threshold:
            return label
    return "CONGESTED"


def get_flow_rate(vehicle_timestamps, window_seconds=60):
    """Vehicles/hour, extrapolated from how many were counted in the last
    `window_seconds` — a rolling rate rather than an all-time average, so it
    reflects CURRENT conditions, not the whole run so far."""
    now = time.time()
    recent = [t for t in vehicle_timestamps if now - t <= window_seconds]
    return round(len(recent) * (3600 / window_seconds), 1)


def get_los(density):
    """HCM (Highway Capacity Manual) Level of Service band for a given
    vehicle density (veh/km)."""
    for threshold, letter, desc in LOS_THRESHOLDS:
        if density <= threshold:
            return letter, desc
    return "F", "Forced/Breakdown"


def get_peak_minute(traffic_log):
    """✅ #22 — the single busiest minute-of-day observed so far this run."""
    if not traffic_log:
        return "N/A"
    minute_counts = defaultdict(int)
    for entry in traffic_log:
        minute_counts[entry["time"][:5]] += 1   # "HH:MM:SS"[:5] -> "HH:MM"
    peak = max(minute_counts, key=minute_counts.get)
    return f"{peak} ({minute_counts[peak]} vehicles)"


def reset_state():
    with state_lock:
        state["running"]     = False
        state["finished"]    = False
        state["counts"]      = {v: {"IN": 0, "OUT": 0} for v in VEHICLE_CLASSES.values()}
        state["speed_log"]   = []
        state["traffic_log"] = []
        state["congestion"]  = "FREE FLOW"
        state["current_on_road"] = 0
        state["flow_rate"]   = 0.0
        state["density"]     = 0.0
        state["los_letter"]  = "A"
        state["los_desc"]    = "Free Flow"
        state["peak_minute"] = "N/A"
        state["fps"]         = 0.0
        state["frame"]       = None
        state["error"]       = None


def run_detection(video_path=None, use_camera=False):
    """Background detection thread — works for both video file and webcam."""
    global camera_frame

    cap = None
    csv_file = None
    loop_error = None

    try:
        # ✅ #4 — check the model file exists BEFORE trying to load it, so a
        # missing best1.pt gives a clear message instead of a raw traceback
        # buried in the background thread (which the user would never see).
        model_full_path = os.path.join(BASE_DIR, MODEL_PATH)
        if not os.path.exists(model_full_path):
            with state_lock:
                state["error"]   = (f"Model file not found: '{MODEL_PATH}'. "
                                    f"Place your trained model in the project folder "
                                    f"(expected at: {model_full_path}).")
                state["running"] = False
            return

        if not use_camera:
            if not video_path or not os.path.exists(video_path):
                with state_lock:
                    state["error"]   = f"Video file not found: '{video_path}'."
                    state["running"] = False
                return

        try:
            model = YOLO(model_full_path)
        except Exception as model_exc:
            with state_lock:
                state["error"]   = f"Could not load model '{MODEL_PATH}': {model_exc}"
                state["running"] = False
            return

        # ── Open video source ─────────────────────────────────────────────────
        if use_camera:
            cap = None
            for idx in range(3):
                test = cv2.VideoCapture(idx)
                if test.isOpened():
                    cap = test
                    break
                test.release()
            if cap is None:
                with state_lock:
                    state["error"]   = "No camera found. Make sure webcam is connected."
                    state["running"] = False
                return
        else:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                with state_lock:
                    state["error"]   = ("Could not open video file — it may be "
                                        "corrupted or in an unsupported format.")
                    state["running"] = False
                return

        if use_camera:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)

        # ── Per-run tracking state ──────────────────────────────────────────────
        track_history         = defaultdict(list)
        counted_ids           = set()   # a track counts once, period — see notes below
        vehicle_label_history = defaultdict(lambda: deque(maxlen=LABEL_HISTORY_SIZE))
        confirmed_labels      = {}
        vehicle_timestamps    = []   # ✅ #20 — every counted vehicle's timestamp, for flow rate
        frame_id              = 0

        # ✅ #14 — FPS tracking (actual processing throughput, not source FPS)
        fps_frame_counter = 0
        fps_timer_start   = time.time()
        last_metrics_update = time.time()

        csv_file = open(CSV_PATH, "w", newline="")
        csv_w    = csv.writer(csv_file)
        csv_w.writerow(["Time", "Vehicle", "Direction",
                         "Speed(km/h)", "Confidence", "Congestion"])

        # Camera mode uses its own light skip for a livelier feed; video mode
        # uses config's FRAME_SKIP (keep at 1 for reliable tracking/counting —
        # see project notes on why frame-skipped tracking breaks ID continuity).
        skip = 2 if use_camera else FRAME_SKIP

        try:
            while True:
                with state_lock:
                    if not state["running"]:
                        break

                ret, raw_frame = cap.read()
    
                if not ret:
                    if use_camera:
                        time.sleep(0.01)
                        continue
                    else:
                        break
    
                frame_id += 1
                if frame_id % skip != 0:
                    continue
    
                # Squash-resize to the SAME dimensions the training images were
                # exported at (Roboflow "stretch to fit") — this is the working
                # frame for detection, tracking, AND drawing.
                frame = cv2.resize(raw_frame, (DETECT_W, DETECT_H))
                h, w, _ = frame.shape
    
                results = model.track(
                    frame, persist=True,
                    conf=0.25, iou=0.45,
                    tracker="custom_bytetrack.yaml",
                    verbose=False
                )
    
                current_vehicles = 0
    
                if results[0].boxes.id is not None:
                    boxes   = results[0].boxes.xyxy.cpu().numpy()
                    ids     = results[0].boxes.id.cpu().numpy()
                    classes = results[0].boxes.cls.cpu().numpy()
                    confs   = results[0].boxes.conf.cpu().numpy()
    
                    current_vehicles = sum(
                        1 for c in classes if int(c) in VEHICLE_CLASSES
                    )
                    frame_area = w * h
    
                    for box, tid, cls, conf in zip(boxes, ids, classes, confs):
                        cls = int(cls)
                        if cls not in VEHICLE_CLASSES:
                            continue
    
                        tid   = int(tid)
                        label = VEHICLE_CLASSES[cls]
    
                        x1, y1, x2, y2 = map(int, box)
                        box_width  = x2 - x1
                        box_height = y2 - y1
                        box_area   = box_width * box_height
                        area_frac  = box_area / frame_area if frame_area > 0 else 0
    
                        if box_width > w * 0.7 or box_height > h * 0.7:
                            continue
    
                        aspect_ratio = box_width / max(box_height, 1)
    
                        # Only reclassify when the model is unsure AND the box is
                        # big enough for shape to mean anything — small/distant
                        # boxes are too imprecisely localized to trust geometry.
                        if conf < HIGH_CONF_TRUST and area_frac >= MIN_AREA_FOR_RECLASSIFICATION:
                            if label == "car":
                                if area_frac > TRUCK_MIN_AREA_FRAC and aspect_ratio < 1.5:
                                    label = "truck"
                                elif area_frac > BUS_MIN_AREA_FRAC and aspect_ratio > 1.6:
                                    label = "bus"
                            if label == "motorbike" and area_frac > TRICYCLE_MIN_AREA_FRAC and aspect_ratio < 1.8:
                                label = "tricycle"
    
                        # Confidence-weighted voting with hysteresis, so a
                        # handful of noisy low-confidence frames can't outvote a
                        # smaller number of high-confidence ones, and the
                        # displayed label doesn't flicker between contenders.
                        vehicle_label_history[tid].append((label, conf))
                        history = vehicle_label_history[tid]
                        weighted_scores = defaultdict(float)
                        for hist_label, hist_conf in history:
                            weighted_scores[hist_label] += hist_conf
                        ranked = sorted(weighted_scores.items(), key=lambda kv: kv[1], reverse=True)
                        top_label, top_score = ranked[0]
    
                        current_confirmed = confirmed_labels.get(tid)
                        if current_confirmed is None:
                            confirmed_labels[tid] = top_label
                        elif top_label != current_confirmed:
                            current_score = weighted_scores.get(current_confirmed, 0.0)
                            if top_score > current_score * LABEL_SWITCH_MARGIN:
                                confirmed_labels[tid] = top_label
    
                        label = confirmed_labels[tid]
                        color = BOX_COLORS.get(label, (255, 255, 255))
    
                        # Bottom-center anchor — tracks the vehicle's road
                        # contact point far more consistently than the centroid,
                        # which shifts around based on box height.
                        cx = (x1 + x2) // 2
                        cy = y2
    
                        track_history[tid].append((cx, cy))
                        if len(track_history[tid]) > 30:
                            track_history[tid].pop(0)
    
                        for px, py in track_history[tid][-8:]:
                            cv2.circle(frame, (px, py), 2, color, -1)
    
                        speed      = estimate_speed(track_history[tid], tid)
                        congestion = get_congestion(current_vehicles)
    
                        # ── Counting: proximity trigger ─────────────────────────
                        # Count a vehicle the first time it lands inside the
                        # zone, deduped by track_id — no persistent side-state
                        # to get stuck in. Direction is worked out from this
                        # track's own trajectory so far.
                        if LINE_UPPER <= cy <= LINE_LOWER and tid not in counted_ids:
                            counted_ids.add(tid)
                            pts = track_history[tid]
                            if len(pts) >= 2:
                                net = pts[-1][1] - pts[0][1]
                                direction = "IN" if net >= 0 else "OUT"
                            else:
                                direction = "IN"
    
                            with state_lock:
                                state["counts"][label][direction] += 1
                                if speed > 0:
                                    state["speed_log"].append(speed)
                                state["traffic_log"].append({
                                    "time":      datetime.now().strftime("%H:%M:%S"),
                                    "vehicle":   label,
                                    "direction": direction,
                                    "speed":     speed
                                })
                            vehicle_timestamps.append(time.time())
                            csv_w.writerow([
                                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                label, direction, speed, f"{conf:.2f}", congestion
                            ])
    
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        txt = f"{label} {conf:.0%} | {speed}km/h"
                        tw  = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
                        cv2.rectangle(frame, (x1, y1 - 18), (x1 + tw + 4, y1), color, -1)
                        cv2.putText(frame, txt, (x1 + 2, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
                        cv2.circle(frame, (cx, cy), 4, color, -1)
    
                # ✅ #14 — FPS: measured over a rolling 1-second window, so it
                # reflects actual processing throughput, not the source
                # video's own frame rate.
                fps_frame_counter += 1
                now_ts = time.time()
                if now_ts - fps_timer_start >= 1.0:
                    current_fps = fps_frame_counter / (now_ts - fps_timer_start)
                    fps_frame_counter = 0
                    fps_timer_start = now_ts
                    with state_lock:
                        state["fps"] = round(current_fps, 1)

                # ✅ #20 — real flow/density/LOS analytics, recomputed once a
                # second (no need to do this every frame — it's a rolling
                # rate, not something that changes meaningfully frame-to-frame).
                if now_ts - last_metrics_update >= 1.0:
                    last_metrics_update = now_ts
                    flow_rate = get_flow_rate(vehicle_timestamps)
                    with state_lock:
                        recent_speeds = state["speed_log"][-50:]
                        avg_speed = (sum(recent_speeds) / len(recent_speeds)
                                    if recent_speeds else 1)
                        density = flow_rate / max(avg_speed, 1)
                        los_letter, los_desc = get_los(density)
                        state["flow_rate"]   = flow_rate
                        state["density"]     = round(density, 1)
                        state["los_letter"]  = los_letter
                        state["los_desc"]    = los_desc
                        state["peak_minute"] = get_peak_minute(state["traffic_log"])

                # ── Zone visual: soft translucent band + IN/OUT arrows ────────────
                # No solid "counting line" — a band with clear directional arrows
                # reads at a glance and matches the rest of the dashboard.
                zone_overlay = frame.copy()
                cv2.rectangle(zone_overlay, (0, LINE_UPPER), (w, LINE_LOWER),
                              (235, 235, 235), -1)
                cv2.addWeighted(zone_overlay, 0.10, frame, 0.90, 0, frame)
                cv2.line(frame, (0, LINE_UPPER), (w, LINE_UPPER), (235, 235, 235), 1)
                cv2.line(frame, (0, LINE_LOWER), (w, LINE_LOWER), (235, 235, 235), 1)
    
                zone_mid_y = (LINE_UPPER + LINE_LOWER) // 2
                cv2.arrowedLine(frame, (50, zone_mid_y - 22), (50, zone_mid_y + 22),
                                IN_COLOR_BGR, 3, tipLength=0.4)
                cv2.putText(frame, "IN", (62, zone_mid_y + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, IN_COLOR_BGR, 2)
                cv2.arrowedLine(frame, (w - 90, zone_mid_y + 22), (w - 90, zone_mid_y - 22),
                                OUT_COLOR_BGR, 3, tipLength=0.4)
                cv2.putText(frame, "OUT", (w - 78, zone_mid_y + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, OUT_COLOR_BGR, 2)

                with state_lock:
                    display_fps = state["fps"]
                cv2.putText(frame, f"FPS: {display_fps}",
                            (10, h - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                cv2.putText(frame, f"ON ROAD: {current_vehicles}",
                            (10, h - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    
                with state_lock:
                    state["congestion"]      = get_congestion(current_vehicles)
                    state["current_on_road"] = current_vehicles
                    _, jpeg = cv2.imencode(".jpg", frame,
                                           [cv2.IMWRITE_JPEG_QUALITY, 60])
                    state["frame"] = jpeg.tobytes()

        except Exception as loop_exc:
            # ✅ A mid-loop error (bad frame, model hiccup, etc.) no longer
            # skips cleanup — we record it and still fall through to close
            # the CSV and attempt graph generation with whatever data was
            # captured before the error, instead of losing everything.
            loop_error = str(loop_exc)

        # ── Cleanup — always runs, whether the loop finished normally,
        # was stopped by the user, or hit an error above ─────────────────────
        cap.release()
        csv_file.close()

        # Generate and copy graphs (video mode only)
        graph_status = None
        if not use_camera:
            graph_status = generate_graphs(csv_path=CSV_PATH, output_dir=GRAPHS_DIR)
            for f in ["traffic_analysis_dashboard.png", "accuracy_report.png"]:
                src = os.path.join(GRAPHS_DIR, f)
                dst = os.path.join(STATIC_GRAPHS_DIR, f)
                if os.path.exists(src):
                    shutil.copy(src, dst)

        with state_lock:
            if loop_error:
                state["error"] = loop_error
            elif graph_status == "empty":
                # Not a crash — the run genuinely completed with zero
                # vehicles ever crossing the counting zone, so there was
                # nothing to plot. Surfaced as info, not an error, so it
                # doesn't look like the app is broken.
                state["error"] = ("No vehicles were counted during this run "
                                  "(nothing crossed the counting zone), so no "
                                  "CSV rows or graphs were generated.")
            state["running"]  = False
            state["finished"] = True

    except Exception as e:
        with state_lock:
            state["error"]   = str(e)
            state["running"] = False


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    reset_state()

    use_camera = request.form.get("use_camera") == "true"

    if use_camera:
        with state_lock:
            state["use_camera"] = True
            state["running"]    = True
        t = threading.Thread(
            target=run_detection, args=(None, True), daemon=True)
        t.start()
        return redirect(url_for("dashboard"))

    if "video" in request.files and request.files["video"].filename:
        video = request.files["video"]
        path  = os.path.join(UPLOAD_DIR, "video.mp4")
        video.save(path)
        with state_lock:
            state["use_camera"] = False
            state["running"]    = True
        t = threading.Thread(
            target=run_detection, args=(path, False), daemon=True)
        t.start()
        return redirect(url_for("dashboard"))

    return redirect(url_for("index"))


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/stop", methods=["POST"])
def stop():
    with state_lock:
        state["running"] = False
    return jsonify({"status": "stopping"})


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            with state_lock:
                frame = state.get("frame")
            if frame:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.04)   # ~25 fps max to browser
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    with state_lock:
        counts    = dict(state["counts"])
        speed_log = list(state["speed_log"])
        logs      = list(state["traffic_log"][-10:])
        running   = state["running"]
        finished  = state["finished"]
        error     = state["error"]
        congestion = state["congestion"]
        current_on_road = state["current_on_road"]
        flow_rate  = state["flow_rate"]
        density    = state["density"]
        los_letter = state["los_letter"]
        los_desc   = state["los_desc"]
        peak_minute = state["peak_minute"]
        fps        = state["fps"]

    grand_total = sum(v["IN"] + v["OUT"] for v in counts.values())
    avg_speed   = (round(sum(speed_log[-50:]) / len(speed_log[-50:]), 1)
                   if speed_log else 0)

    return jsonify({
        "running":     running,
        "finished":    finished,
        "error":       error,
        "congestion":  congestion,
        "grand_total": grand_total,
        "avg_speed":   avg_speed,
        "counts":      counts,
        "logs":        logs,
        # ✅ #20/#21/#22/#14
        "current_on_road": current_on_road,
        "flow_rate":   flow_rate,
        "density":     density,
        "los_letter":  los_letter,
        "los_desc":    los_desc,
        "peak_minute": peak_minute,
        "fps":         fps,
    })


@app.route("/results")
def results():
    with state_lock:
        counts    = dict(state["counts"])
        speed_log = list(state["speed_log"])
        logs      = list(state["traffic_log"])
        peak_minute = state["peak_minute"]

    grand_total = sum(v["IN"] + v["OUT"] for v in counts.values())
    avg_speed   = (round(sum(speed_log) / len(speed_log), 1)
                   if speed_log else 0)
    return render_template("results.html",
                           counts=counts,
                           grand_total=grand_total,
                           avg_speed=avg_speed,
                           logs=logs,
                           peak_minute=peak_minute,
                           # ✅ Cache-busting value — without this, browsers
                           # cache the graph PNGs by URL, so a second
                           # detection run can keep showing the OLD graph
                           # image even after it's been overwritten on disk.
                           cache_bust=int(time.time()))


@app.route("/download_csv")
def download_csv():
    if os.path.exists(CSV_PATH):
        return send_file(CSV_PATH,
                         as_attachment=True,
                         download_name="traffic_report.csv")
    return "No data yet", 404


port = int(os.environ.get("PORT", 5000))
   debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
   app.run(debug=debug_mode, threaded=True, host="0.0.0.0", port=port)
