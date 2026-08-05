import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tkinter as tk

here = os.path.dirname(os.path.abspath(__file__))
savePath = os.path.join(here, "policy.pth")
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

GRID = 7
NIN, NACT, HIDDEN = 28, 3, 256
LR, GAMMA, LAM = 3e-4, 0.99, 0.95
CLIP, ENT_COEF, VF_COEF = 0.2, 0.01, 0.5
EPOCHS, BATCH, ROLLOUT = 4, 256, 2048
MAX_STEPS, HUNGER = 500, GRID * GRID
UPDATES = 1000
PLAY_SPEED = 100

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
STEP = {UP: (0, -1), RIGHT: (1, 0), DOWN: (0, 1), LEFT: (-1, 0)}
SIDE = {UP: (1, 0), RIGHT: (0, 1), DOWN: (-1, 0), LEFT: (0, -1)}
TURN_LEFT = {UP: LEFT, LEFT: DOWN, DOWN: RIGHT, RIGHT: UP}
TURN_RIGHT = {UP: RIGHT, RIGHT: DOWN, DOWN: LEFT, LEFT: UP}
RAYS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]

RAY_NAMES = ["S", "FR", "R", "BR", "B", "BL", "L", "FL"]
IN_NAMES = [RAY_NAMES[i // 3] + "wbf"[i % 3] for i in range(24)] + ["fF", "fR", "ln", "rch"]
OUT_NAMES = ["<L", "^S", "R>"]


class Snake:
    def __init__(self):
        self.reset()

    def reset(self):
        c = GRID // 2
        self.body = [(c, c), (c - 1, c), (c - 2, c)]
        self.dir = RIGHT
        self.steps = self.apples = self.hunger = 0
        self.limit = MAX_STEPS
        self.food = None
        self.place()
        return self.obs()

    def place(self):
        free = [(x, y) for x in range(GRID) for y in range(GRID) if (x, y) not in self.body]
        self.food = random.choice(free) if free else None

    def openArea(self):
        wall = set(self.body)
        seen = set()
        hx, hy = self.body[0]
        stack = [(hx + dx, hy + dy) for dx, dy in STEP.values()]
        while stack:
            x, y = stack.pop()
            if (x, y) in seen or (x, y) in wall or x < 0 or x >= GRID or y < 0 or y >= GRID:
                continue
            seen.add((x, y))
            stack += [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return len(seen)

    def obs(self):
        hx, hy = self.body[0]
        body = set(self.body[1:])
        fx, fy = STEP[self.dir]
        rx, ry = SIDE[self.dir]
        out = []

        for a, b in RAYS:
            dx, dy = a * fx + b * rx, a * fy + b * ry
            wall = hit = food = 0.0
            x, y, dist = hx, hy, 0
            while True:
                x += dx
                y += dy
                dist += 1
                if x < 0 or x >= GRID or y < 0 or y >= GRID:
                    wall = 1.0 / dist
                    break
                if not hit and (x, y) in body:
                    hit = 1.0 / dist
                if self.food and (x, y) == self.food:
                    food = 1.0
            out += [wall, hit, food]

        gx, gy = self.food if self.food else (hx, hy)
        vx, vy = gx - hx, gy - hy
        out += [
            (vx * fx + vy * fy) / GRID,
            (vx * rx + vy * ry) / GRID,
            len(self.body) / (GRID * GRID),
            self.openArea() / (GRID * GRID),
        ]
        return out

    def step(self, a):
        ox, oy = self.body[0]
        if a == 0:
            self.dir = TURN_LEFT[self.dir]
        elif a == 2:
            self.dir = TURN_RIGHT[self.dir]

        dx, dy = STEP[self.dir]
        nx, ny = ox + dx, oy + dy
        if nx < 0 or nx >= GRID or ny < 0 or ny >= GRID or (nx, ny) in self.body[1:]:
            return self.obs(), -1.0, True

        self.body.insert(0, (nx, ny))
        self.steps += 1
        self.hunger += 1

        if (nx, ny) == self.food:
            self.apples += 1
            self.hunger = 0
            self.limit = MAX_STEPS + self.apples * 50
            self.place()
            r = 1.0
        else:
            self.body.pop()
            gx, gy = self.food if self.food else self.body[0]
            old = abs(ox - gx) + abs(oy - gy)
            new = abs(nx - gx) + abs(ny - gy)
            r = 0.01 if new < old else -0.01

        done = self.steps >= self.limit or self.hunger > HUNGER
        if self.hunger > HUNGER:
            r = -0.5
        return self.obs(), r, done


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = nn.Sequential(nn.Linear(NIN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN), nn.Tanh())
        self.pi = nn.Linear(HIDDEN, NACT)
        self.v = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        h = self.core(x)
        return self.pi(h), self.v(h).squeeze(-1)

    def act(self, o):
        logits, val = self(torch.tensor(o, dtype=torch.float32, device=device).unsqueeze(0))
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return a.item(), dist.log_prob(a).item(), val.item()


def gray(v):
    c = max(0, min(255, int(v * 255)))
    return f"#{c:02x}{c:02x}{c:02x}"


def drawNet(cv, net, rewards, apples, upd, w=1200, h=720):
    cv.delete("all")
    leftW = int(w * 0.52)
    chartX = int(w * 0.55)
    margin = 28
    k = 12

    w1 = net.core[0].weight.detach().cpu().float().numpy()
    w2 = net.core[2].weight.detach().cpu().float().numpy()
    wp = net.pi.weight.detach().cpu().float().numpy()
    h1 = np.argsort(np.abs(w1).sum(1))[-k:]
    h2 = np.argsort(np.abs(w2[:, h1]).sum(1))[-k:]
    w1, w2, wp = w1[h1], w2[np.ix_(h2, h1)], wp[:, h2]
    for a in (w1, w2, wp):
        m = np.abs(a).max()
        if m:
            a /= m

    xs = [int(leftW * v) for v in [0.13, 0.38, 0.63, 0.88]]

    def y(n, i):
        return margin + int((h - 2 * margin) * (i + 1) / (n + 1))

    for j in range(k):
        for i in range(NIN):
            m = abs(w1[j, i])
            if m >= 0.15:
                cv.create_line(xs[0], y(NIN, i), xs[1], y(k, j),
                               fill=gray(m * (0.75 if w1[j, i] > 0 else 0.28)), width=max(1, int(m * 1.5)))

    for j in range(k):
        for i in range(k):
            m = abs(w2[j, i])
            if m >= 0.18:
                cv.create_line(xs[1], y(k, i), xs[2], y(k, j),
                               fill=gray(m * (0.8 if w2[j, i] > 0 else 0.3)), width=max(1, int(m * 2)))

    for j in range(NACT):
        for i in range(k):
            m = abs(wp[j, i])
            if m >= 0.12:
                cv.create_line(xs[2], y(k, i), xs[3], y(NACT, j),
                               fill=gray(m * (0.9 if wp[j, i] > 0 else 0.35)), width=max(1, int(m * 2.5)))

    for i in range(NIN):
        yy = y(NIN, i)
        cv.create_oval(xs[0] - 3, yy - 3, xs[0] + 3, yy + 3, fill="#444444", outline="")
        cv.create_text(xs[0] - 6, yy, text=IN_NAMES[i], fill="#333333", font=("Courier", 6), anchor="e")

    for i in range(k):
        for x in xs[1:3]:
            yy = y(k, i)
            cv.create_oval(x - 5, yy - 5, x + 5, yy + 5, fill="#555555", outline="")

    for i in range(NACT):
        yy = y(NACT, i)
        cv.create_oval(xs[3] - 7, yy - 7, xs[3] + 7, yy + 7, fill="#ffffff", outline="")
        cv.create_text(xs[3] + 10, yy, text=OUT_NAMES[i], fill="#ffffff", font=("Courier", 9), anchor="w")

    cv.create_text(leftW // 2, 10, text=f"network  upd {upd}", fill="#444444", font=("Courier", 8))

    ch = (h - 60) // 2
    for ci, (hist, name) in enumerate([(rewards, "reward"), (apples, "apples")]):
        y0 = 20 + ci * (ch + 18)
        y1 = y0 + ch
        cw = w - chartX - 20
        cv.create_rectangle(chartX, y0, w - 10, y1, fill="#080808", outline="#1e1e1e")
        cv.create_text(chartX + 6, y0 + 5, text=name, fill="#444444", font=("Courier", 7), anchor="nw")
        if len(hist) > 1:
            lo, hi = min(hist), max(hist)
            rng = hi - lo or 1e-9
            pts = []
            for i, val in enumerate(hist):
                pts += [chartX + 8 + int((cw - 16) * i / max(len(hist) - 1, 1)),
                        y1 - 6 - int((ch - 18) * (val - lo) / rng)]
            cv.create_line(pts, fill="#ffffff", width=1)
            cv.create_text(w - 14, y0 + 5, text=f"{hist[-1]:.1f}", fill="#666666", font=("Courier", 7), anchor="ne")


def train(updates=UPDATES, viz=False):
    print("device:", device)
    net = Net().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR)

    if os.path.exists(savePath):
        net.load_state_dict(torch.load(savePath, map_location=device, weights_only=True))
        print("resumed")

    rewards, apples = [], []
    root = cv = None
    w, h = 1200, 720

    if viz:
        root = tk.Tk()
        root.title("PPO Snake - training")
        root.configure(bg="#000000")
        cv = tk.Canvas(root, width=w, height=h, bg="#000000", highlightthickness=0)
        cv.pack()
        root.update()

    env = Snake()

    for upd in range(1, updates + 1):
        obs, acts, logps, vals, rews, dones = [], [], [], [], [], []
        epReward, epRewards, epApples = 0.0, [], []
        o = env.reset()

        net.eval()
        with torch.no_grad():
            for _ in range(ROLLOUT):
                a, lp, v = net.act(o)
                no, r, done = env.step(a)
                obs.append(o)
                acts.append(a)
                logps.append(lp)
                vals.append(v)
                rews.append(r)
                dones.append(float(done))
                epReward += r
                o = no
                if done:
                    epRewards.append(epReward)
                    epApples.append(env.apples)
                    epReward = 0.0
                    o = env.reset()
            lastV = net.act(o)[2] if not dones[-1] else 0.0

        obs = np.array(obs, dtype=np.float32)
        vals = np.array(vals, dtype=np.float32)
        rews = np.array(rews, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        adv = np.zeros(len(rews), dtype=np.float32)
        gae = 0.0
        for t in range(len(rews) - 1, -1, -1):
            nextV = (lastV if t == len(rews) - 1 else vals[t + 1]) * (1.0 - dones[t])
            delta = rews[t] + GAMMA * nextV - vals[t]
            gae = delta + GAMMA * LAM * (1.0 - dones[t]) * gae
            adv[t] = gae
        ret = adv + vals

        x = torch.tensor(obs, device=device)
        a = torch.tensor(np.array(acts), device=device)
        oldLogp = torch.tensor(np.array(logps, dtype=np.float32), device=device)
        adv = torch.tensor(adv, device=device)
        ret = torch.tensor(ret, device=device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        ids = np.arange(len(obs))
        net.train()
        for _ in range(EPOCHS):
            np.random.shuffle(ids)
            for s in range(0, len(ids), BATCH):
                b = ids[s:s + BATCH]
                logits, val = net(x[b])
                dist = torch.distributions.Categorical(logits=logits)
                ratio = torch.exp(dist.log_prob(a[b]) - oldLogp[b])
                piLoss = -torch.min(ratio * adv[b], torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * adv[b]).mean()
                loss = piLoss + VF_COEF * F.mse_loss(val, ret[b]) - ENT_COEF * dist.entropy().mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()

        meanR = np.mean(epRewards) if epRewards else 0.0
        meanA = np.mean(epApples) if epApples else 0.0
        rewards.append(meanR)
        apples.append(meanA)
        print(f"upd {upd:4d} | eps {len(epRewards):3d} | meanRew {meanR:7.2f} | meanApples {meanA:5.1f}")

        if viz and cv:
            drawNet(cv, net, rewards, apples, upd, w, h)
            root.update()

        if upd % 3 == 0:
            torch.save(net.state_dict(), savePath)

    torch.save(net.state_dict(), savePath)
    print("done")
    if viz and root:
        root.mainloop()


def play(speed=PLAY_SPEED):
    if not os.path.exists(savePath):
        print("no policy, run: python main.py train")
        return

    net = Net().to(device)
    net.load_state_dict(torch.load(savePath, map_location=device, weights_only=True))
    net.eval()

    cell = 36
    side = GRID * cell
    env = Snake()
    obs = [env.reset()]
    live = [True]

    root = tk.Tk()
    root.title("PPO Snake")
    root.configure(bg="#000000")
    cv = tk.Canvas(root, width=side + 160, height=side, bg="#000000", highlightthickness=0)
    cv.pack()

    def draw():
        cv.delete("all")
        for x in range(GRID):
            for y in range(GRID):
                cv.create_rectangle(x * cell, y * cell, x * cell + cell, y * cell + cell,
                                    fill="#0a0a0a", outline="#111111", width=1)
        if env.food:
            x, y = env.food
            cv.create_rectangle(x * cell + 2, y * cell + 2, x * cell + cell - 2, y * cell + cell - 2,
                                fill="#cc2222", outline="")
        for i, (x, y) in enumerate(env.body):
            cv.create_rectangle(x * cell + 2, y * cell + 2, x * cell + cell - 2, y * cell + cell - 2,
                                fill="#44cc44" if i == 0 else "#226622", outline="")
        x = side + 18
        mono = ("Courier", 10)
        cv.create_text(x, 20, text="PPO SNAKE", fill="#ffffff", font=("Courier", 11, "bold"), anchor="w")
        cv.create_text(x, 52, text=f"apples  {env.apples:>4}", fill="#ffffff", font=mono, anchor="w")
        cv.create_text(x, 72, text=f"steps   {env.steps:>4}", fill="#444444", font=mono, anchor="w")
        cv.create_text(x, side - 16, text="alive" if live[0] else "dead",
                       fill="#ffffff" if live[0] else "#555555", font=mono, anchor="w")

    def tick():
        if live[0]:
            with torch.no_grad():
                a = net.act(obs[0])[0]
            obs[0], _, done = env.step(a)
            live[0] = not done
        else:
            obs[0] = env.reset()
            live[0] = True
        draw()
        root.after(speed, tick)

    tick()
    root.mainloop()


mode = sys.argv[1] if len(sys.argv) > 1 else "train"
if mode == "play":
    play()
else:
    train(viz="--viz" in sys.argv)
