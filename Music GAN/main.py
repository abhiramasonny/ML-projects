import os
import sys
import math
import random
from pathlib import Path
from collections import defaultdict
import mido
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

here = os.path.dirname(os.path.abspath(__file__))
runDir = os.path.join(here, "runs")
genDir = os.path.join(here, "generated")
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

PAD, BOS, EOS, BAR = 0, 1, 2, 3
PITCH_OFF, VEL_OFF, TIME_OFF = 4, 132, 164
VOCAB = 196

SEQ_LEN = 128
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 3
FFN_DIM = 256
DROP = 0.1
BATCH = 32
LR_G, LR_D = 2e-4, 2e-4
LAMBDA_GP = 10
N_CRITIC = 2
COND_LEN = 32
MARKOV_N = 2
EPOCHS = 100
TEMP = 1.0
NUM_GENERATE = 4


def midiToTokens(path):
    try:
        mid = mido.MidiFile(path)
    except Exception:
        return []

    ticksPerBeat = mid.ticks_per_beat
    events = []
    for track in mid.tracks:
        t = 0
        for msg in track:
            t += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                events.append((t, msg.note, msg.velocity))
    events.sort(key=lambda e: e[0])

    tokens = [BOS]
    prev = 0
    barLen = ticksPerBeat * 4
    nextBar = barLen
    for t, note, vel in events:
        while t >= nextBar:
            tokens.append(BAR)
            nextBar += barLen
        if t - prev > 0:
            tokens.append(TIME_OFF + min(31, int((t - prev) * 32 / (ticksPerBeat + 1e-6))))
        tokens.append(VEL_OFF + min(31, vel // 4))
        tokens.append(PITCH_OFF + note)
        prev = t
    tokens.append(EOS)
    return tokens


def tokensToMidi(tokens, path, ticksPerBeat=480):
    mid = mido.MidiFile(ticks_per_beat=ticksPerBeat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    vel = 64

    for tok in tokens:
        if tok == BAR:
            continue
        if TIME_OFF <= tok < TIME_OFF + 32:
            continue
        if VEL_OFF <= tok < VEL_OFF + 32:
            vel = (tok - VEL_OFF + 1) * 4
        elif PITCH_OFF <= tok < PITCH_OFF + 128:
            note = tok - PITCH_OFF
            track.append(mido.Message("note_on", note=note, velocity=vel, time=0))
            track.append(mido.Message("note_off", note=note, velocity=0, time=ticksPerBeat // 2))

    mid.save(str(path))


class Markov:
    def __init__(self, n=MARKOV_N):
        self.n = n
        self.trans = defaultdict(lambda: defaultdict(int))
        self.states = []

    def fit(self, sequences):
        for seq in sequences:
            pitches = [t - PITCH_OFF for t in seq if PITCH_OFF <= t < PITCH_OFF + 128]
            for i in range(len(pitches) - self.n):
                self.trans[tuple(pitches[i:i + self.n])][pitches[i + self.n]] += 1
        self.states = list(self.trans.keys())

    def sample(self, length):
        if not self.states:
            return [random.randint(48, 72) for _ in range(length)]
        state = random.choice(self.states)
        out = list(state)
        while len(out) < length:
            options = self.trans.get(state)
            if not options:
                state = random.choice(self.states)
                out.extend(state)
                continue
            total = sum(options.values())
            r = random.random() * total
            acc = 0
            nxt = next(iter(options))
            for k, v in options.items():
                acc += v
                if r <= acc:
                    nxt = k
                    break
            out.append(nxt)
            state = tuple(out[-self.n:])
        return [max(0, min(127, p)) for p in out[:length]]

    def save(self):
        return {"n": self.n, "trans": {k: dict(v) for k, v in self.trans.items()}}

    def load(self, data):
        self.n = data["n"]
        self.trans = defaultdict(lambda: defaultdict(int))
        for state, options in data["trans"].items():
            for pitch, count in options.items():
                self.trans[state][pitch] = count
        self.states = list(self.trans.keys())


class MidiDataset(Dataset):
    def __init__(self, root, seqLen=SEQ_LEN):
        self.seqs = []
        paths = list(Path(root).rglob("*.mid")) + list(Path(root).rglob("*.midi"))
        for p in paths:
            tokens = midiToTokens(str(p))
            for i in range(0, max(0, len(tokens) - seqLen), seqLen // 2):
                chunk = tokens[i:i + seqLen]
                if len(chunk) == seqLen:
                    self.seqs.append(chunk)
        print(f"dataset: {len(self.seqs)} sequences from {len(paths)} files")

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return torch.tensor(self.seqs[i], dtype=torch.long)


class PosEmb(nn.Module):
    def __init__(self, d, maxLen=4096):
        super().__init__()
        pe = torch.zeros(maxLen, d)
        pos = torch.arange(maxLen).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class SelfAttnBlock(nn.Module):
    def __init__(self, d, heads, ffn, drop):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d))
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        h = self.n1(x)
        x = x + self.drop(self.attn(h, h, h)[0])
        return x + self.drop(self.ff(self.n2(x)))


class CrossAttnBlock(nn.Module):
    def __init__(self, d, heads, ffn, drop):
        super().__init__()
        self.self1 = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.cross = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d))
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)
        self.n3 = nn.LayerNorm(d)
        self.drop = nn.Dropout(drop)

    def forward(self, x, kv):
        h = self.n1(x)
        x = x + self.drop(self.self1(h, h, h)[0])
        x = x + self.drop(self.cross(self.n2(x), kv, kv)[0])
        return x + self.drop(self.ff(self.n3(x)))


class Generator(nn.Module):
    def __init__(self, d=D_MODEL, heads=N_HEADS, layers=N_LAYERS, ffn=FFN_DIM, seqLen=SEQ_LEN):
        super().__init__()
        self.seqLen = seqLen
        self.d = d
        self.condEmb = nn.Embedding(129, d)
        self.condPos = PosEmb(d)
        self.queryPos = PosEmb(d)
        self.layers = nn.ModuleList([CrossAttnBlock(d, heads, ffn, DROP) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB)

    def forward(self, cond, temp=1.0):
        kv = self.condPos(self.condEmb(cond))
        x = self.queryPos(torch.randn(cond.size(0), self.seqLen, self.d, device=cond.device))
        for layer in self.layers:
            x = layer(x, kv)
        logits = self.head(self.norm(x))
        return logits, F.gumbel_softmax(logits, tau=max(temp, 0.1), hard=False)


class Discriminator(nn.Module):
    def __init__(self, d=D_MODEL, heads=N_HEADS, layers=N_LAYERS, ffn=FFN_DIM):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d, padding_idx=PAD)
        self.pos = PosEmb(d)
        self.layers = nn.ModuleList([SelfAttnBlock(d, heads, ffn, DROP) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))

    def fromEmb(self, emb):
        x = self.pos(emb)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x).mean(1)).squeeze(-1)

    def forward(self, tokens):
        return self.fromEmb(self.emb(tokens))


def train(dataPath):
    os.makedirs(runDir, exist_ok=True)
    print("device:", device)

    dataset = MidiDataset(dataPath)
    if len(dataset) == 0:
        print("no sequences found - point at a folder of .mid files")
        return
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True, drop_last=True)

    markov = Markov()
    markov.fit(dataset.seqs)
    print(f"markov chain: {len(markov.states)} states")

    G = Generator().to(device)
    D = Discriminator().to(device)
    optG = torch.optim.Adam(G.parameters(), lr=LR_G, betas=(0.0, 0.9))
    optD = torch.optim.Adam(D.parameters(), lr=LR_D, betas=(0.0, 0.9))

    for ep in range(1, EPOCHS + 1):
        bar = tqdm(loader, desc=f"ep {ep:3d}/{EPOCHS}", unit="batch")
        for real in bar:
            real = real.to(device)
            n = real.size(0)

            for _ in range(N_CRITIC):
                cond = torch.tensor([markov.sample(COND_LEN) for _ in range(n)],
                                    dtype=torch.long, device=device)
                with torch.no_grad():
                    softTokens = G(cond, TEMP)[1]
                fakeEmb = softTokens @ D.emb.weight
                realEmb = D.emb(real)

                alpha = torch.rand(n, 1, 1, device=device)
                interp = (alpha * realEmb.detach() + (1 - alpha) * fakeEmb.detach()).requires_grad_(True)
                score = D.fromEmb(interp)
                grads = torch.autograd.grad(score, interp, torch.ones_like(score), create_graph=True)[0]
                gp = ((grads.norm(2, dim=[1, 2]) - 1) ** 2).mean()

                lossD = D.fromEmb(fakeEmb).mean() - D.fromEmb(realEmb).mean() + LAMBDA_GP * gp
                optD.zero_grad()
                lossD.backward()
                optD.step()

            cond = torch.tensor([markov.sample(COND_LEN) for _ in range(n)],
                                dtype=torch.long, device=device)
            softTokens = G(cond, TEMP)[1]
            lossG = -D.fromEmb(softTokens @ D.emb.weight).mean()
            optG.zero_grad()
            lossG.backward()
            optG.step()

            bar.set_postfix(D=f"{lossD.item():.3f}", G=f"{lossG.item():.3f}")

        torch.save({"G": G.state_dict(), "D": D.state_dict(), "markov": markov.save()},
                   os.path.join(runDir, "model.pt"))
    print("saved", os.path.join(runDir, "model.pt"))


def generate(modelPath):
    os.makedirs(genDir, exist_ok=True)
    ckpt = torch.load(modelPath, map_location=device, weights_only=False)

    G = Generator().to(device)
    G.load_state_dict(ckpt["G"])
    G.eval()
    markov = Markov()
    markov.load(ckpt["markov"])

    for i in range(NUM_GENERATE):
        cond = torch.tensor([markov.sample(COND_LEN)], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = G(cond, TEMP)[0]
        path = os.path.join(genDir, f"gen_{i + 1:03d}.mid")
        tokensToMidi([BOS] + logits.argmax(-1).squeeze(0).tolist(), path)
        print("saved", path)


mode = sys.argv[1] if len(sys.argv) > 1 else "generate"
if mode == "train":
    train(sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "data"))
else:
    generate(sys.argv[2] if len(sys.argv) > 2 else os.path.join(runDir, "model.pt"))
