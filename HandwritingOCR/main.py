import os
import sys
import csv
import json
import tarfile
import urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from pylatexenc.latex2text import LatexNodes2Text

pwd = os.path.dirname(os.path.abspath(__file__))
data = os.path.join(pwd, "data")
cachePath = os.path.join(data, "cache.npz")
classesPath = os.path.join(data, "classes.json")
modelPath = os.path.join(data, "model.pth")
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 32
EPOCHS, BATCH, LR = 25, 256, 1e-3
CANVAS_SIZE = 320
PEN_WIDTH = 14
DATASET_URL = "https://zenodo.org/records/259444/files/HASYv2.tar.bz2"

augment = transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=5, fill=0)


class Net(nn.Module):
    def __init__(self, numClasses):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3, padding=1)
        self.b1 = nn.BatchNorm2d(32)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.b2 = nn.BatchNorm2d(64)
        self.c3 = nn.Conv2d(64, 128, 3, padding=1)
        self.b3 = nn.BatchNorm2d(128)
        self.c4 = nn.Conv2d(128, 256, 3, padding=1)
        self.b4 = nn.BatchNorm2d(256)
        self.drop = nn.Dropout(0.4)
        self.fc1 = nn.Linear(256 * 2 * 2, 512)
        self.b5 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, numClasses)

    def forward(self, t):
        t = F.max_pool2d(F.relu(self.b1(self.c1(t))), 2)
        t = F.max_pool2d(F.relu(self.b2(self.c2(t))), 2)
        t = F.max_pool2d(F.relu(self.b3(self.c3(t))), 2)
        t = F.max_pool2d(F.relu(self.b4(self.c4(t))), 2)
        t = t.flatten(1)
        t = self.drop(F.relu(self.b5(self.fc1(t))))
        return self.fc2(t)


def binarize(arr):
    ink = (arr.astype(np.float32) > 127.0).astype(np.float32)
    return 1.0 - ink if ink.mean() > 0.5 else ink


def prep():
    os.makedirs(data, exist_ok=True)
    archive = os.path.join(data, "hasy.tar.bz2")

    if not os.path.exists(archive):
        print("downloading HASYv2 ...")
        urllib.request.urlretrieve(DATASET_URL, archive)
    if not os.path.exists(os.path.join(data, "hasy-data-labels.csv")):
        print("extracting ...")
        with tarfile.open(archive, "r:bz2") as tf:
            tf.extractall(data)

    with open(os.path.join(data, "hasy-data-labels.csv"), newline="") as f:
        rows = list(csv.DictReader(f))

    classes = sorted(set(r["latex"] for r in rows))
    indexByLatex = {label: i for i, label in enumerate(classes)}
    images, targets = [], []

    for n, row in enumerate(rows):
        path = os.path.join(data, row["path"])
        if not os.path.exists(path):
            path = os.path.join(data, os.path.basename(row["path"]))
        if not os.path.exists(path):
            continue
        im = Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE))
        images.append(binarize(np.asarray(im)).astype(np.uint8))
        targets.append(indexByLatex[row["latex"]])
        if n % 10000 == 0:
            print("processed", n, "/", len(rows))

    x = np.stack(images).astype(np.uint8)
    y = np.asarray(targets, dtype=np.int64)
    np.savez_compressed(cachePath, x=x, y=y)
    with open(classesPath, "w") as f:
        json.dump(classes, f)
    print("cache built:", x.shape, "classes:", len(classes))


def train():
    if not os.path.exists(cachePath):
        prep()

    blob = np.load(cachePath)
    with open(classesPath) as f:
        classes = json.load(f)

    x, y = blob["x"], blob["y"]
    print("device:", device, "samples:", len(x), "classes:", len(classes))

    perm = np.random.permutation(len(x))
    x, y = x[perm], y[perm]
    cut = int(len(x) * 0.9)
    xTrain = torch.from_numpy(x[:cut]).float().unsqueeze(1)
    yTrain = torch.from_numpy(y[:cut])
    xTest = torch.from_numpy(x[cut:]).float().unsqueeze(1).to(device)
    yTest = torch.from_numpy(y[cut:]).to(device)

    net = Net(len(classes)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    lossFn = nn.CrossEntropyLoss(label_smoothing=0.1)
    steps = int(np.ceil(len(xTrain) / BATCH))

    for ep in range(EPOCHS):
        net.train()
        order = torch.randperm(len(xTrain))
        running = 0.0
        for s in range(steps):
            idx = order[s * BATCH:(s + 1) * BATCH]
            loss = lossFn(net(augment(xTrain[idx]).to(device)), yTrain[idx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
        sch.step()

        net.eval()
        with torch.no_grad():
            acc = (net(xTest).argmax(1) == yTest).float().mean().item()
        print(f"epoch {ep + 1} loss {running / steps:.4f} valAcc {acc:.4f}")

    torch.save(net.state_dict(), modelPath)
    print("saved", modelPath)


def draw():
    if not os.path.exists(modelPath):
        print("train model first: uv run main.py train")
        return

    import tkinter as tk
    from PIL import ImageDraw

    with open(classesPath) as f:
        classes = json.load(f)

    net = Net(len(classes)).to(device)
    net.load_state_dict(torch.load(modelPath, map_location=device, weights_only=True))
    net.eval()
    converter = LatexNodes2Text()

    root = tk.Tk()
    root.title("Handwriting Recognition")
    canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="white", cursor="cross")
    canvas.grid(row=0, column=0, rowspan=8, padx=8, pady=8)
    surface = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
    pen = ImageDraw.Draw(surface)
    last = {"x": None, "y": None}

    title = tk.Label(root, text="draw a symbol", font=("Helvetica", 16, "bold"))
    title.grid(row=0, column=1, sticky="w", padx=8)
    rows = []
    for r in range(5):
        label = tk.Label(root, text="", font=("Helvetica", 14), anchor="w", width=24)
        label.grid(row=r + 1, column=1, sticky="w", padx=8)
        rows.append(label)

    def toSymbol(latex):
        try:
            return converter.latex_to_text(latex)
        except Exception:
            return latex

    def onDown(ev):
        last["x"], last["y"] = ev.x, ev.y

    def onMove(ev):
        if last["x"] is None:
            last["x"], last["y"] = ev.x, ev.y
        canvas.create_line(last["x"], last["y"], ev.x, ev.y, width=PEN_WIDTH,
                           fill="black", capstyle=tk.ROUND, smooth=True)
        pen.line([last["x"], last["y"], ev.x, ev.y], fill=255, width=PEN_WIDTH)
        last["x"], last["y"] = ev.x, ev.y

    def onUp(ev):
        last["x"], last["y"] = None, None

    def clear():
        canvas.delete("all")
        pen.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=0)
        title.config(text="draw a symbol")
        for label in rows:
            label.config(text="")

    def tick():
        arr = np.asarray(surface)
        if arr.max() >= 1:
            rowsHit = np.where(np.any(arr > 0, axis=1))[0]
            colsHit = np.where(np.any(arr > 0, axis=0))[0]
            pad = max(rowsHit[-1] - rowsHit[0], colsHit[-1] - colsHit[0]) // 8 + 4
            rMin, rMax = max(0, rowsHit[0] - pad), min(arr.shape[0] - 1, rowsHit[-1] + pad)
            cMin, cMax = max(0, colsHit[0] - pad), min(arr.shape[1] - 1, colsHit[-1] + pad)

            cropped = surface.crop((cMin, rMin, cMax + 1, rMax + 1))
            w, h = cropped.size
            side = max(w, h)
            square = Image.new("L", (side, side), 0)
            square.paste(cropped, ((side - w) // 2, (side - h) // 2))

            grid = binarize(np.asarray(square.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)))
            t = torch.from_numpy(grid).float().unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = F.softmax(net(t), 1)[0].cpu().numpy()

            top5 = [(classes[i], float(probs[i])) for i in probs.argsort()[::-1][:5]]
            title.config(text="predicting: " + toSymbol(top5[0][0]))
            for i, (latex, p) in enumerate(top5):
                rows[i].config(text=f"{i + 1}.  {toSymbol(latex)}   {p * 100:.1f}%")

        root.after(1000, tick)

    canvas.bind("<Button-1>", onDown)
    canvas.bind("<B1-Motion>", onMove)
    canvas.bind("<ButtonRelease-1>", onUp)
    tk.Button(root, text="Clear", command=clear, width=12).grid(row=7, column=1, sticky="w", padx=8)
    tick()
    root.mainloop()


mode = sys.argv[1] if len(sys.argv) > 1 else "draw"

if mode == "prep":
    prep()
elif mode == "train":
    train()
else:
    draw()
