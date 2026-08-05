import os
import sys
import glob
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne

here = os.path.dirname(os.path.abspath(__file__))
rawDir = os.path.join(here, "data", "raw")
epochPath = os.path.join(here, "data", "processed", "epochs.npz")
outDir = os.path.join(here, "outputs")
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

CHANNEL = "EEG Fpz-Cz"
EPOCH_SECONDS = 30.0
SUBJECTS = 20
RECORDINGS = [1]
EPOCHS, BATCH, LR, SEED = 40, 128, 1e-3, 42
STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]
ANNOTATION_MAP = {"Sleep stage W": 1, "Sleep stage 1": 2, "Sleep stage 2": 3,
                  "Sleep stage 3": 4, "Sleep stage 4": 4, "Sleep stage R": 5}
EVENT_IDS = {"W": 1, "N1": 2, "N2": 3, "N3": 4, "REM": 5}


class SleepCNN(nn.Module):
    def __init__(self, numClasses=5, sampleRate=100, drop=0.5):
        super().__init__()
        kernel, stride = sampleRate // 2, max(sampleRate // 16, 1)
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel, stride, kernel // 2, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(True), nn.MaxPool1d(8, 8), nn.Dropout(drop),
            nn.Conv1d(64, 128, 8, 1, 4, bias=False), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Conv1d(128, 128, 8, 1, 4, bias=False), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Conv1d(128, 128, 8, 1, 4, bias=False), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.MaxPool1d(4, 4), nn.Dropout(drop))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(drop), nn.Linear(128, numClasses))

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.classifier(self.pool(self.features(x)))


def download():
    os.makedirs(rawDir, exist_ok=True)
    paths = mne.datasets.sleep_physionet.age.fetch_data(
        subjects=list(range(SUBJECTS)), recording=RECORDINGS, path=rawDir, on_missing="warn")
    print(f"downloaded {len(paths)} pairs -> {rawDir}")


def preprocess():
    psgFiles = sorted(glob.glob(os.path.join(rawDir, "**", "*-PSG.edf"), recursive=True))
    hypFiles = glob.glob(os.path.join(rawDir, "**", "*-Hypnogram.edf"), recursive=True)
    if not psgFiles:
        raise SystemExit(f"no data in {rawDir}, run: uv run main.py download")

    allX, allY, allSubjects = [], [], []

    for psg in psgFiles:
        prefix = os.path.basename(psg)[:6]
        matches = [h for h in hypFiles if os.path.basename(h).startswith(prefix)]
        if not matches:
            continue

        raw = mne.io.read_raw_edf(psg, stim_channel=None, verbose="ERROR")
        if CHANNEL not in raw.ch_names:
            continue
        raw.pick_channels([CHANNEL])

        annotations = mne.read_annotations(matches[0])
        raw.set_annotations(annotations, emit_warning=False)
        annotations.crop(annotations[1]["onset"] - 1800, annotations[-2]["onset"] + 1800)
        raw.set_annotations(annotations, emit_warning=False)

        sampleRate = raw.info["sfreq"]
        try:
            events, _ = mne.events_from_annotations(raw, event_id=ANNOTATION_MAP,
                                                    chunk_duration=EPOCH_SECONDS, verbose="ERROR")
        except ValueError:
            continue

        epochs = mne.Epochs(raw, events, event_id=EVENT_IDS, tmin=0, tmax=EPOCH_SECONDS - 1 / sampleRate,
                            baseline=None, preload=True, verbose="ERROR")
        x = epochs.get_data(picks=CHANNEL).astype(np.float32)[:, 0]
        y = (epochs.events[:, 2] - 1).astype(np.int64)

        width = int(round(EPOCH_SECONDS * sampleRate))
        if x.shape[1] != width:
            x = x[:, :width]
        if len(x) == 0:
            continue

        try:
            subjectId = int(os.path.basename(psg)[3:5])
        except ValueError:
            subjectId = -1

        allX.append(x)
        allY.append(y)
        allSubjects.append(np.full(len(y), subjectId, np.int64))

    X = np.concatenate(allX)
    y = np.concatenate(allY)
    subjects = np.concatenate(allSubjects)
    os.makedirs(os.path.dirname(epochPath), exist_ok=True)
    np.savez_compressed(epochPath, X=X, y=y, subject=subjects, class_names=np.array(STAGE_NAMES))
    print(f"saved {len(X)} epochs -> {epochPath}")


def train():
    os.makedirs(outDir, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    blob = np.load(epochPath, allow_pickle=True)
    X, y, subjects = blob["X"], blob["y"], blob["subject"]
    names = [str(c) for c in blob["class_names"]]

    unique = np.unique(subjects)
    rng = np.random.default_rng(SEED)
    rng.shuffle(unique)
    nTest = max(1, int(round(len(unique) * 0.15)))
    nVal = max(1, int(round(len(unique) * 0.15)))
    testIds, valIds, trainIds = set(unique[:nTest]), set(unique[nTest:nTest + nVal]), set(unique[nTest + nVal:])
    trainMask = np.array([s in trainIds for s in subjects])
    valMask = np.array([s in valIds for s in subjects])
    testMask = np.array([s in testIds for s in subjects])

    Xn = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-7)

    def loader(mask, shuffle):
        return DataLoader(TensorDataset(torch.from_numpy(Xn[mask]).float(),
                                        torch.from_numpy(y[mask]).long()), BATCH, shuffle)

    trainLoader = loader(trainMask, True)
    valLoader = loader(valMask, False)
    testLoader = loader(testMask, False)

    counts = np.bincount(y[trainMask], minlength=len(names)).astype(np.float64)
    weights = torch.tensor(counts.sum() / (len(names) * np.maximum(counts, 1)),
                           dtype=torch.float32, device=device)

    model = SleepCNN(len(names)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(weight=weights)

    def predict(source):
        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for xb, yb in source:
                preds.append(model(xb.to(device)).argmax(1).cpu().numpy())
                truths.append(yb.numpy())
        return np.concatenate(preds), np.concatenate(truths)

    losses, accs, f1s = [], [], []
    bestF1, bestState = -1.0, None

    for ep in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for xb, yb in trainLoader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
        sch.step()

        valPred, valTrue = predict(valLoader)
        acc = accuracy_score(valTrue, valPred)
        f1 = f1_score(valTrue, valPred, average="macro")
        losses.append(running / len(trainLoader.dataset))
        accs.append(acc)
        f1s.append(f1)
        print(f"epoch {ep}/{EPOCHS} loss {losses[-1]:.4f} val_acc {acc:.3f} val_f1 {f1:.3f}")

        if f1 > bestF1:
            bestF1 = f1
            bestState = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if bestState:
        model.load_state_dict(bestState)
    torch.save(model.state_dict(), os.path.join(outDir, "sleep_cnn.pth"))

    testPred, testTrue = predict(testLoader)
    acc = accuracy_score(testTrue, testPred)
    macroF1 = f1_score(testTrue, testPred, average="macro")
    kappa = cohen_kappa_score(testTrue, testPred)
    report = classification_report(testTrue, testPred, target_names=names,
                                   labels=list(range(len(names))), digits=3, zero_division=0)
    print(f"test acc {acc:.3f} mf1 {macroF1:.3f} kappa {kappa:.3f}\n{report}")

    with open(os.path.join(outDir, "metrics.json"), "w") as f:
        json.dump({"test_accuracy": acc, "test_macro_f1": macroF1, "test_cohen_kappa": kappa,
                   "best_val_macro_f1": bestF1, "test_subjects": sorted(int(s) for s in testIds),
                   "n_train_epochs": int(trainMask.sum()), "n_test_epochs": int(testMask.sum())}, f, indent=2)
    with open(os.path.join(outDir, "classification_report.txt"), "w") as f:
        f.write(report)

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    steps = range(1, len(losses) + 1)
    ax1.plot(steps, losses, "tab:red")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss", color="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(steps, accs, "tab:blue")
    ax2.plot(steps, f1s, "tab:green")
    ax2.set_ylabel("val")
    fig.tight_layout()
    fig.savefig(os.path.join(outDir, "training_curves.png"), dpi=130)
    plt.close(fig)

    matrix = confusion_matrix(testTrue, testPred, labels=list(range(len(names))))
    normed = matrix / np.maximum(matrix.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(normed, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{normed[i, j]:.2f}", ha="center", va="center",
                    color="w" if normed[i, j] > 0.5 else "k", fontsize=9)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(outDir, "confusion_matrix.png"), dpi=130)
    plt.close(fig)

    testSubjects = subjects[testMask]
    nightMask = testSubjects == np.unique(testSubjects)[0]
    with torch.no_grad():
        night = model(torch.from_numpy(Xn[testMask][nightMask]).float().to(device)).argmax(1).cpu().numpy()
    hours = np.arange(len(night)) * 0.5 / 60
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.step(hours, y[testMask][nightMask], where="post", color="k", lw=1.2, label="expert")
    ax.step(hours, night, where="post", color="tab:orange", lw=1, alpha=0.8, label="pred")
    ax.set_xlabel("hours")
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outDir, "example_hypnogram.png"), dpi=130)
    plt.close(fig)

    print("saved ->", outDir)


mode = sys.argv[1] if len(sys.argv) > 1 else "all"
if mode == "download":
    download()
elif mode == "preprocess":
    preprocess()
elif mode == "train":
    train()
else:
    download()
    preprocess()
    train()
