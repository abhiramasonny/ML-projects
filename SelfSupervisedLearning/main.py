import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

here = os.path.dirname(os.path.abspath(__file__))
dataDir = os.path.join(here, "data")
backbonePath = os.path.join(here, "backbone.pth")
probePath = os.path.join(here, "probe.pth")
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 10
PROJ_DIM = 128
TAU = 0.5
TRAIN_EPOCHS, TRAIN_BATCH, TRAIN_LR = 20, 512, 3e-4
PROBE_EPOCHS, PROBE_BATCH, PROBE_LR = 50, 256, 0.01
TSNE_SAMPLES = 2000
CLASS_NAMES = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

augment = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    normalize,
])


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=None)
        base.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        base.maxpool = nn.Identity()
        self.enc = nn.Sequential(*list(base.children())[:-1])
        self.proj = nn.Sequential(
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, PROJ_DIM),
        )

    def encode(self, x):
        return self.enc(x).flatten(1)

    def forward(self, x):
        return self.proj(self.encode(x))


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, label = self.base[i]
        return augment(img), augment(img), label


def loadBackbone():
    net = Net().to(device)
    net.load_state_dict(torch.load(backbonePath, map_location=device, weights_only=True))
    net.eval()
    return net


def train():
    print("device:", device)
    raw = datasets.CIFAR10(dataDir, train=True, download=True)
    loader = torch.utils.data.DataLoader(PairDataset(raw), batch_size=TRAIN_BATCH,
                                         shuffle=True, num_workers=0, drop_last=True)
    net = Net().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=TRAIN_LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TRAIN_EPOCHS)

    for ep in tqdm(range(TRAIN_EPOCHS), desc="epochs"):
        net.train()
        running = 0.0
        bar = tqdm(loader, desc=f"train {ep + 1}/{TRAIN_EPOCHS}", leave=False)

        for x1, x2, _ in bar:
            z = F.normalize(torch.cat([net(x1.to(device)), net(x2.to(device))]), dim=1)
            n = x1.shape[0]
            sim = z @ z.T / TAU
            sim.fill_diagonal_(float("-inf"))
            labels = torch.cat([torch.arange(n, 2 * n), torch.arange(n)]).to(device)
            loss = F.cross_entropy(sim, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}")

        sch.step()
        if ep % 10 == 0 or ep == TRAIN_EPOCHS - 1:
            tqdm.write(f"epoch {ep + 1} loss {running / len(loader):.4f}")

    torch.save(net.state_dict(), backbonePath)
    print("saved", backbonePath)


def probe():
    trainSet = datasets.CIFAR10(dataDir, train=True, download=True, transform=normalize)
    testSet = datasets.CIFAR10(dataDir, train=False, download=True, transform=normalize)
    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=PROBE_BATCH, shuffle=True, num_workers=0)
    testLoader = torch.utils.data.DataLoader(testSet, batch_size=PROBE_BATCH, shuffle=False, num_workers=0)

    net = loadBackbone()
    for p in net.parameters():
        p.requires_grad_(False)

    head = nn.Linear(512, NUM_CLASSES).to(device)
    opt = torch.optim.SGD(head.parameters(), lr=PROBE_LR, momentum=0.9, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=PROBE_EPOCHS)

    for ep in tqdm(range(PROBE_EPOCHS), desc="epochs"):
        head.train()
        running = 0.0
        bar = tqdm(trainLoader, desc=f"probe {ep + 1}/{PROBE_EPOCHS}", leave=False)

        for x, y in bar:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                feat = net.encode(x)
            loss = F.cross_entropy(head(feat), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}")

        sch.step()

        if ep % 10 == 0 or ep == PROBE_EPOCHS - 1:
            head.eval()
            correct = total = 0
            with torch.no_grad():
                for x, y in tqdm(testLoader, desc="eval", leave=False):
                    x, y = x.to(device), y.to(device)
                    correct += (head(net.encode(x)).argmax(1) == y).sum().item()
                    total += y.size(0)
            tqdm.write(f"epoch {ep + 1} train loss {running / len(trainLoader):.4f} "
                       f"test acc {correct / total * 100:.2f}%")

    torch.save(head.state_dict(), probePath)
    print("saved", probePath)


def viz():
    testSet = datasets.CIFAR10(dataDir, train=False, download=True, transform=normalize)
    loader = torch.utils.data.DataLoader(testSet, batch_size=256, shuffle=False, num_workers=0)
    net = loadBackbone()

    feats, labels, seen = [], [], 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc="extracting embeddings"):
            feats.append(net.encode(x.to(device)).cpu())
            labels.append(y)
            seen += len(y)
            if seen >= TSNE_SAMPLES:
                break

    feats = torch.cat(feats)[:TSNE_SAMPLES].numpy()
    labels = torch.cat(labels)[:TSNE_SAMPLES].numpy()

    print("running t-SNE ...")
    emb = TSNE(n_components=2, perplexity=40, random_state=0, verbose=1).fit_transform(feats)

    fig, ax = plt.subplots(figsize=(9, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))
    for c in range(NUM_CLASSES):
        mask = labels == c
        ax.scatter(emb[mask, 0], emb[mask, 1], color=colors[c], label=CLASS_NAMES[c], s=8, alpha=0.7)
    ax.legend(markerscale=3, loc="best")
    ax.set_title("t-SNE of SimCLR embeddings - CIFAR-10")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


mode = sys.argv[1] if len(sys.argv) > 1 else "train"
if mode == "probe":
    probe()
elif mode == "viz":
    viz()
else:
    train()
