## Football Player Tracking & Re‑Identification

A Python-based pipeline for detecting, tracking, and re-identifying football players (or other person-like objects) in video streams. It combines YOLO object detection, ByteTrack multi-object tracking, and OSNet person re-identification to assign consistent IDs to players across frames.

---

### 🔍 Features

* **Accurate object detection** using YOLOv11 (ultralytics).
* **Robust multi-object tracking** with ByteTrack algorithm.
* **Person re-identification (ReID)** via OSNet model (torchreid) to maintain consistent identities across occlusions.
* **Customizable gallery**: catalog feature embeddings and match new detections with a threshold.
* **Color-coded bounding boxes** for different classes (players, goalkeeper, ball, referees).
* **Export annotated video** with labeled bounding boxes and IDs.

---

### 🛠️ Requirements

* Python 3.11.4
* CUDA-enabled GPU (recommended) or CPU fallback


---

### 📥 Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/KunalNath04/Re-id_initial.git
   cd Re-id_initial
   ```

2. **Create and activate a virtual environment** (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\\Scripts\\activate  # Windows
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Download weights**:

   * Place your YOLO model weights (`best.pt`) in `models/`
   * Place your input video (`15sec_input_720p.mp4`) in `assets/`
   * TorchReID will automatically download OSNet weights on first run.

---

### 🚀 Usage

Run the pipeline on a video file and generate an output with annotated tracks:

```bash
python object_tracking_reid.py
```

* Output is saved to the path specified by `OUTPUT_FILE`.
* Progress and final confirmation printed in console.

---

### ⚙️ Configuration

* **Paths & parameters** are set at the top of `object_tracking_reid.py`:

  ```python
  YOLO_WEIGHTS   = "models/best.pt"
  VIDEO_FILE     = "assets/15sec_input_720p.mp4"
  TRACKER_CONFIG = "bytetrack.yaml"
  OUTPUT_FILE    = "output_reid_fixed.mp4"

  REID_THRESH    = 0.8           # gallery matching threshold
  CROP_SIZE      = (128, 256)    # width, height for ReID crops
  REID_CLASSES   = {"player"}    # classes to apply ReID
  ```

* **ByteTrack settings** are in `bytetrack.yaml`:

  ```yaml
  tracker_type: bytetrack
  track_high_thresh: 0.25
  track_low_thresh: 0.1
  new_track_thresh: 0.6
  track_buffer: 30
  match_thresh: 0.8
  fuse_score: False
  ```

* **Color scheme** in `object_tracking_reid.py`:

  ```python
  COLORS = {
      'ball':       (0, 255, 255),   # cyan
      'goalkeeper': (255,   0,   0),  # red
      'player':     (  0, 255,   0),  # green
      'referee':    (  0,   0, 255),  # blue
  }
  ```

---

### 📂 File Structure

```
├── assets/                      # Input videos, sample clips
│   └── 15sec_input_720p.mp4
│
├── models/                      # YOLO model weights
│   └── best.pt
│
├── configs/                     # Tracking configs
│   └── bytetrack.yaml
│
├── gallery.py                   # PersonGallery: stores & matches embeddings
├── reid_model.py                # ReidModel: feature extractor (OSNet)
├── object_tracking_reid.py      # Main tracking & ReID script
├── test.py                      # Quick test for torchreid imports
└── requirements.txt
```


### 📝 How It Works

1. **Detection & Tracking**

   * Load YOLOv11 detector and ByteTrack tracker.
   * For each frame: detect objects, associate detections across frames, assign ByteTrack IDs.

2. **Crop & Embedding**

   * For detections of class `player`, crop the bounding box region.
   * Resize to `CROP_SIZE` and feed to OSNet to extract feature embeddings.

3. **Gallery Matching**

   * Maintain a gallery of registered embeddings with unique UIDs.
   * For each new embedding, compute cosine similarity to gallery means.
   * If similarity ≥ `REID_THRESH`, reuse UID; else, register new UID.

4. **Visualization**

   * Draw colored bounding boxes and labels (`<class> <UID>`).
   * Non-player classes show `<class>_<trackID>`.

---

### 🔧 Customization

* **Thresholds & sizes**: adjust `REID_THRESH` and `CROP_SIZE` for different domains.
* **Classes for ReID**: modify `REID_CLASSES` to include additional person-like classes.
* **Color scheme**: change `COLORS` mapping for your preferences.
* **Tracker parameters**: tune `bytetrack.yaml` for detection/tracking sensitivity.

---




*Happy tracking!*
