import cv2
import numpy as np
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

weights = hf_hub_download(
    repo_id="Kota0612/okra11n-seg-v5",
    filename="output/okra_finetune_v5/weights/best.pt",
)
model = YOLO(weights)

results = model.predict(source="image.png", imgsz=640, conf=0.25)

for r in results:
    if r.masks is not None:
        for i, mask in enumerate(r.masks.xy):  # polygon points per detected okra
            poly = np.asarray(mask, dtype=np.float32)
            if len(poly) < 3:
                continue  # degenerate polygon, skip

            m = cv2.moments(poly)
            if m["m00"] < 1e-6:
                continue  # zero-area polygon, avoid divide-by-zero

            cx = m["m10"] / m["m00"]
            cy = m["m01"] / m["m00"]

            conf = float(r.boxes.conf[i])
            print(f"Okra {i}: centroid=({cx:.1f}, {cy:.1f}) conf={conf:.2f}")