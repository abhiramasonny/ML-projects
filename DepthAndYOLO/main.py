import cv2
import numpy as np
import torch
from ultralytics import YOLO

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

CAMERA = 0
YOLO_MODEL = "yolo11m.pt"
DEPTH_SKIP = 3
ALPHA = 0.55
COLORMAP = cv2.COLORMAP_INFERNO

print("device:", device)
print("loading YOLO11m")
yolo = YOLO(YOLO_MODEL)
print("loading MiDaS v3 Small")
depthNet = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(device).eval()
print("press q to quit")

cap = cv2.VideoCapture(CAMERA)
if not cap.isOpened():
    raise SystemExit("cannot open camera")

depth = None
tick = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    if tick % DEPTH_SKIP == 0:
        h, w = frame.shape[:2]
        img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (384, 384))
        img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        with torch.no_grad():
            raw = depthNet(img).squeeze().cpu().numpy().astype(np.float32)
        raw = cv2.resize(raw, (w, h))
        depth = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    tick += 1

    if depth is not None:
        colored = cv2.applyColorMap(((1 - depth) * 255).astype(np.uint8), COLORMAP)
        canvas = cv2.addWeighted(frame, 1 - ALPHA, colored, ALPHA, 0)
    else:
        canvas = frame.copy()

    for box in yolo(frame, verbose=False)[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        name = yolo.names[int(box.cls[0])]

        d = 0.0
        if depth is not None:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            hw, hh = max(1, (x2 - x1) // 4), max(1, (y2 - y1) // 4)
            roi = depth[max(0, cy - hh):cy + hh, max(0, cx - hw):cx + hw]
            d = float(np.median(roi)) if roi.size else 0.0

        label = f"{name} {float(box.conf[0]):.2f}  d:{d:.2f}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 1)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
        cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 0, 0), -1)
        cv2.putText(canvas, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_DUPLEX, 0.45, (255, 255, 255), 1)

    cv2.imshow("Depth + YOLO", canvas)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
