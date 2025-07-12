import os
# Suppress duplicate MKL warnings and TF logs
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["YOLO_LOG_LEVEL"]      = "ERROR"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import numpy as np
from ultralytics import YOLO

from reid_model import ReidModel
from gallery     import PersonGallery

# ——— Paths & params ———
YOLO_WEIGHTS   = "models/best.pt"
VIDEO_FILE     = "assets/15sec_input_720p.mp4"
TRACKER_CONFIG = "bytetrack.yaml"
OUTPUT_FILE    = "output_reid_fixed.mp4"

REID_THRESH    = 0.8           # match threshold in PersonGallery
CROP_SIZE      = (128, 256)    # width, height for ReidModel.extract()
REID_CLASSES   = {"player"}    # only run Re‑ID on players

COLORS = {
    'ball':       (0, 255, 255),
    'goalkeeper': (255,   0,   0),
    'player':     (  0, 255,   0),
    'referee':    (  0,   0, 255),
}
THICKNESS = 2

class TrackingWithReID:
    def __init__(self):
        # load YOLOv11 + ByteTrack
        self.yolo = YOLO(YOLO_WEIGHTS)
        self.yolo.overrides["verbose"] = False
        self.yolo.overrides["show"]    = False

        self.names = self.yolo.names

        # ReID + gallery
        self.reid    = ReidModel(device='cuda')
        self.gallery = PersonGallery(threshold=REID_THRESH)

        # map ByteTrack IDs → gallery UIDs
        self.bt2uid = {}

    def run(self, source=VIDEO_FILE, output=OUTPUT_FILE):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise IOError(f"Cannot open video {source}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out    = cv2.VideoWriter(output, fourcc, fps, (W, H))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # 1) detect + track
            results = self.yolo.track(
                source=frame,
                persist=True,
                tracker=TRACKER_CONFIG,
                verbose=False,
                show=False
            )
            det = results[0]
            if det.boxes is None:
                out.write(frame)
                continue

            boxes   = det.boxes.xyxy.cpu().numpy().astype(int)
            classes = det.boxes.cls.cpu().numpy().astype(int)
            # Safely extract tracker IDs
            raw_ids = getattr(det.boxes, 'id', None)
            if raw_ids is None:
                tids = np.full(len(boxes), -1, dtype=int)
            else:
                tids = raw_ids.cpu().numpy().astype(int)

            # 2) gather crops & keep mapping to detection idx
            crops    = []
            crop_map = []
            for i, cls in enumerate(classes):
                name = self.names[int(cls)]
                if name in REID_CLASSES:
                    x1, y1, x2, y2 = boxes[i]
                    y1c, y2c = max(0, y1), min(H, y2)
                    x1c, x2c = max(0, x1), min(W, x2)
                    crop = frame[y1c:y2c, x1c:x2c]
                    if crop.size == 0:
                        crop = np.zeros((CROP_SIZE[1], CROP_SIZE[0], 3), np.uint8)
                    else:
                        crop = cv2.resize(crop, CROP_SIZE)
                    crops.append(crop)
                    crop_map.append(i)

            # 3) compute features + assign UIDs
            uids = [None] * len(crops)
            if crops:
                feats = self.reid.extract(crops)  # NxD array
                for idx, feat in enumerate(feats):
                    det_i = crop_map[idx]
                    bt_id = int(tids[det_i])

                    if bt_id in self.bt2uid:
                        # reuse existing UID
                        uid = self.bt2uid[bt_id]
                    else:
                        # always register new UID for unique BTID
                        uid = self.gallery.register(feat, frame_idx)
                        self.bt2uid[bt_id] = uid

                    uids[idx] = uid

            # 4) draw boxes & labels
            p = 0
            for (x1,y1,x2,y2), cls, bt_id in zip(boxes, classes, tids):
                name  = self.names[int(cls)]
                color = COLORS.get(name, (200,200,200))

                if name in REID_CLASSES:
                    label = f"{name.capitalize()} {uids[p]}"
                    p += 1
                else:
                    label = f"{name.capitalize()}_{bt_id}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, THICKNESS)
                cv2.putText(
                    frame, label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    THICKNESS
                )

            out.write(frame)

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"✅  Output saved to {output}")

if __name__ == "__main__":
    TrackingWithReID().run()
