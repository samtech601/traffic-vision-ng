# TrafficVision NG

An AI-based vehicle detection, classification, tracking, and traffic-flow analysis system built for Nigerian road traffic. Detects **cars, tricycles (keke), buses, vans, trucks, and motorbikes (okada)**, tracks them across frames, counts them bidirectionally (IN/OUT), estimates speed, and computes real traffic-engineering metrics (flow rate, density, and HCM Level of Service).

Two ways to run it:
- **Desktop app** (`main.py`) — OpenCV window, processes a local video file, saves an annotated output video + CSV log + analysis graphs.
- **Web app** (`app.py`) — Flask app with video upload or live camera detection, a live dashboard, and a downloadable report.

---

## Features

- Custom-trained YOLO object detection, tuned specifically for Nigerian vehicle types
- Multi-object tracking (ByteTrack) with tuned thresholds for occlusion-heavy traffic
- Confidence-weighted classification voting with hysteresis, to keep the displayed vehicle type stable instead of flickering
- Zone-based bidirectional vehicle counting (IN/OUT), robust to frame-skipping
- Real-time speed estimation (calibratable per camera)
- Traffic-flow analytics: flow rate (veh/hr), density (veh/km), and Level of Service per the Highway Capacity Manual (HCM)
- Live FPS and "vehicles currently on road" readout
- Peak-traffic-minute detection
- Auto-generated traffic analysis graphs and downloadable CSV report
- Color-coded bounding boxes and a clean IN/OUT zone overlay (no obtrusive single counting line)

---

## Project Structure

```
TrafficVision/
├── app.py                  # Flask web application
├── main.py                 # Standalone desktop application
├── config.py                # All tunable constants in one place
├── speed.py                 # Speed estimation logic
├── plot.py                  # Graph generation from output.csv
├── hard_negative_miner.py   # Tool for collecting misclassified detections for retraining
├── custom_bytetrack.yaml    # Tuned ByteTrack tracker config
├── requirements.txt
├── templates/                # Flask HTML templates
│   ├── base.html
│   ├── index.html
│   ├── dashboard.html
│   └── results.html
├── static/graphs/            # Generated graph images (web app)
├── graphs/                   # Generated graph images (desktop app)
├── uploads/                  # Uploaded videos (web app)
├── best1.pt                  # ⚠️ Your trained model — NOT included, add this yourself
└── video.mp4                 # ⚠️ Your test video — NOT included, add this yourself
```

---

## Requirements

- Python 3.9+
- A trained YOLO model file (`best1.pt`) matching the vehicle classes below

Install dependencies:
```bash
pip install -r requirements.txt
```

`requirements.txt` includes: `flask`, `ultralytics`, `opencv-python-headless`, `pandas`, `matplotlib`, `numpy`, `gunicorn`.

---

## Setup

1. **Add your trained model.** Place your `best1.pt` file in the project root.
2. **Verify the class mapping matches your model.** Run:
   ```python
   from ultralytics import YOLO
   print(YOLO("best1.pt").names)
   ```
   Compare the output against `vehicle_classes` in `main.py` / `VEHICLE_CLASSES` in `app.py`. These **must** match exactly, or classification will be wrong from the start — this was the single most common bug encountered while building this project.
3. **Add a test video** (for the desktop app, name it `video.mp4` or update `VIDEO_PATH` in `config.py`).
4. **Calibrate speed for your camera.** `PIXEL_TO_METER` in `config.py` converts pixel movement to real-world distance — this depends entirely on your camera's height, angle, and distance from the road. The default value is a starting point, not a universal constant; calibrate it against a known real-world distance in your footage.
5. **Check the counting zone.** `LINE_POSITION` and `OFFSET` in `config.py` define where vehicles are counted. Make sure this zone sits somewhere vehicles are reliably detected on both sides of it.

---

## Usage

### Desktop app
```bash
python main.py
```
Controls: `P` = pause, `S` = screenshot, `R` = reset counts, `Q`/`Esc` = quit. On exit, it prints a summary report, writes `output.csv`, and saves graphs to `graphs/`.

### Web app
```bash
python app.py
```
Then open `http://localhost:5000` in a browser. Upload a video or use a connected camera, watch the live dashboard, and download the CSV/view graphs from the Results page once detection finishes.

---

## Dataset & Model Training

The detection model was custom-trained (not a generic pretrained model) specifically on Nigerian road vehicle types, using [Roboflow](https://roboflow.com) for annotation and [Google Colab](https://colab.research.google.com) for training. Key lessons from the training process (see `hard_negative_miner.py` for the tool built to support this):

- **Class balance matters a lot** — underrepresented classes (originally motorbike, tricycle, van) were the most frequently misclassified.
- **Small/distant vehicles are the hardest case** — a minimum-box-size guard (`MIN_AREA_FOR_RECLASSIFICATION` in `config.py`) prevents unreliable geometry-based heuristics from firing on tiny, imprecisely-localized detections.
- **Bus and van remain separate classes.** An earlier version of this project experimented with merging them after observing confusion between van, bus, and truck — that merge was reverted; bus and van are kept distinct. If you see persistent confusion between these classes, `hard_negative_miner.py` is the tool to use for targeted retraining data rather than merging classes as a workaround.
- Use `hard_negative_miner.py` to collect confidently-wrong predictions from your own footage for targeted retraining:
  ```bash
  python hard_negative_miner.py --video your_video.mp4 --model best1.pt --classes truck bus van --conf 0.6
  ```

---

## Configuration Reference (`config.py`)

| Constant | Purpose |
|---|---|
| `LINE_POSITION`, `OFFSET` | Define the counting zone position and width |
| `FRAME_SKIP` | Frames to skip between detections — keep at `1` for reliable tracking/counting |
| `PIXEL_TO_METER` | Speed calibration — must be set per camera setup |
| `DETECT_W`, `DETECT_H` | Working resolution for detection (matches training preprocessing) |
| `BUS_MIN_AREA_FRAC`, `TRUCK_MIN_AREA_FRAC`, `TRICYCLE_MIN_AREA_FRAC` | Classification heuristic thresholds — recalibrate if you retrain the model |
| `HIGH_CONF_TRUST` | Confidence above which the raw model prediction is trusted outright |
| `MIN_AREA_FOR_RECLASSIFICATION` | Minimum box size before the shape heuristic is trusted |
| `LABEL_SWITCH_MARGIN` | How much a challenger label must "win by" before the displayed label switches |
| `CONGESTION_THRESHOLDS`, `LOS_THRESHOLDS` | Traffic congestion / Level of Service bands |
| `MODEL_PATH`, `VIDEO_PATH` | Model and default video file paths |

---

## Known Limitations

- Classification accuracy depends heavily on the underlying trained model — this system's post-processing (voting, heuristics, hysteresis) improves stability but cannot fully compensate for an undertrained or imbalanced model.
- Speed estimation requires manual per-camera calibration; it is not automatically accurate out of the box.
- Automatic number plate recognition (ANPR) is not implemented — identified as a possible future extension.
- Designed for a single fixed camera angle per run; no multi-camera fusion.

---

## Credits

Built as a final year project combining computer vision (YOLO, ByteTrack) with traffic engineering principles (HCM Level of Service methodology).
