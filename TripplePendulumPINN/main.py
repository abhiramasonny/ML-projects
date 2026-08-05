import os
import sys
import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

here = os.path.dirname(os.path.abspath(__file__))
modelPath = os.path.join(here, "pinn.pth")
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

GRAVITY = 9.81
WINDOW = 2.0
ANGLE_RANGE = 1.0
MASSES = np.array([1.0, 1.0, 1.0])
LENGTHS = np.array([1.0, 1.0, 1.0])
ITERS, BATCH, LR = 15000, 2048, 1e-3
NUM_TRAJECTORIES, TRAJECTORY_STEPS = 400, 200
START_ANGLES = [0.8, -0.5, 0.7]

alphaNp = np.array([[MASSES[max(i, j):].sum() for j in range(3)] for i in range(3)])
betaNp = np.array([MASSES[i:].sum() for i in range(3)])


class Pinn(nn.Module):
    def __init__(self, width=256, bands=8):
        super().__init__()
        self.register_buffer("freqs", torch.arange(1, bands + 1, dtype=torch.float32) * torch.pi)
        layers = [nn.Linear(2 * bands + 3, width), nn.Tanh()]
        for _ in range(4):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 3)]
        self.net = nn.Sequential(*layers)

    def forward(self, t, start):
        tn = t / WINDOW
        ang = tn * self.freqs
        return start + tn * tn * self.net(torch.cat([torch.sin(ang), torch.cos(ang), start], dim=1))


def integrate(start, tGrid):
    def derivative(state):
        theta, vel = state[:3], state[3:]
        diff = theta[:, None] - theta[None, :]
        massMat = alphaNp * np.outer(LENGTHS, LENGTHS) * np.cos(diff)
        coriolis = (alphaNp * np.outer(LENGTHS, LENGTHS) * np.sin(diff) * (vel[None, :] ** 2)).sum(1)
        gravity = betaNp * GRAVITY * LENGTHS * np.sin(theta)
        return np.concatenate([vel, np.linalg.solve(massMat, -(coriolis + gravity))])

    sol = solve_ivp(lambda t, s: derivative(s), (tGrid[0], tGrid[-1]),
                    np.concatenate([start, np.zeros(3)]), t_eval=tGrid, rtol=1e-9, atol=1e-9)
    return sol.y[:3].T


def train():
    print("device:", device, "generating", NUM_TRAJECTORIES, "RK45 trajectories ...")
    tGrid = np.linspace(0, WINDOW, TRAJECTORY_STEPS)
    starts = (np.random.rand(NUM_TRAJECTORIES, 3) * 2 - 1) * ANGLE_RANGE
    data = np.stack([integrate(s.astype(np.float32), tGrid) for s in starts])

    anglesT = torch.tensor(data, dtype=torch.float32, device=device)
    startsT = torch.tensor(starts, dtype=torch.float32, device=device)
    tGridT = torch.tensor(tGrid, dtype=torch.float32, device=device)
    lengths = torch.tensor(LENGTHS, dtype=torch.float32, device=device)
    lenOuter = torch.outer(lengths, lengths)
    alpha = torch.tensor(alphaNp, dtype=torch.float32, device=device)
    beta = torch.tensor(betaNp, dtype=torch.float32, device=device)

    net = Pinn().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ITERS)

    for it in range(ITERS):
        icIdx = torch.randint(0, NUM_TRAJECTORIES, (BATCH,), device=device)
        tIdx = torch.randint(0, TRAJECTORY_STEPS, (BATCH,), device=device)
        pred = net(tGridT[tIdx].unsqueeze(1), startsT[icIdx])
        dataLoss = ((pred - anglesT[icIdx, tIdx]) ** 2).mean()

        t = (torch.rand(BATCH, 1, device=device) * WINDOW).requires_grad_(True)
        start = (torch.rand(BATCH, 3, device=device) * 2 - 1) * ANGLE_RANGE
        theta = net(t, start)
        vel = torch.stack([torch.autograd.grad(theta[:, k].sum(), t, create_graph=True)[0][:, 0]
                           for k in range(3)], 1)
        acc = torch.stack([torch.autograd.grad(vel[:, k].sum(), t, create_graph=True)[0][:, 0]
                           for k in range(3)], 1)

        diff = theta.unsqueeze(2) - theta.unsqueeze(1)
        massMat = alpha.unsqueeze(0) * lenOuter.unsqueeze(0) * torch.cos(diff)
        inertial = torch.bmm(massMat, acc.unsqueeze(2)).squeeze(2)
        coriolis = (alpha.unsqueeze(0) * lenOuter.unsqueeze(0) * torch.sin(diff) * (vel.unsqueeze(1) ** 2)).sum(2)
        gravity = beta.unsqueeze(0) * GRAVITY * lengths.unsqueeze(0) * torch.sin(theta)
        physLoss = ((inertial + coriolis + gravity) ** 2).mean()

        loss = dataLoss + 0.05 * physLoss
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()

        if it % 500 == 0:
            print(f"iter {it} data {dataLoss.item():.6f} phys {physLoss.item():.4f}")

    torch.save(net.state_dict(), modelPath)
    print("saved", modelPath)


def run(start):
    if not os.path.exists(modelPath):
        print("no model yet, run: uv run main.py train")
        return

    net = Pinn().to(device)
    net.load_state_dict(torch.load(modelPath, map_location=device, weights_only=True))
    net.eval()

    steps = 300
    tGrid = np.linspace(0, WINDOW, steps)
    startArr = np.array(start, dtype=np.float32)
    tT = torch.tensor(tGrid, dtype=torch.float32, device=device).unsqueeze(1)
    sT = torch.tensor(startArr, device=device).unsqueeze(0).repeat(steps, 1)

    with torch.no_grad():
        pinnAngles = net(tT, sT).cpu().numpy()
    trueAngles = integrate(startArr, tGrid)
    print("mean abs angle error (rad):", round(np.abs(pinnAngles - trueAngles).mean(), 4))

    px = np.concatenate([np.zeros((steps, 1)), np.cumsum(LENGTHS * np.sin(pinnAngles), 1)], 1)
    py = -np.concatenate([np.zeros((steps, 1)), np.cumsum(LENGTHS * np.cos(pinnAngles), 1)], 1)
    tx = np.concatenate([np.zeros((steps, 1)), np.cumsum(LENGTHS * np.sin(trueAngles), 1)], 1)
    ty = -np.concatenate([np.zeros((steps, 1)), np.cumsum(LENGTHS * np.cos(trueAngles), 1)], 1)

    figAnim, ax = plt.subplots(figsize=(6, 6))
    lim = LENGTHS.sum() * 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_title("triple pendulum: RK45 truth vs PINN")
    trueLine, = ax.plot([], [], "o-", lw=3, color="#1f77b4", label="RK45 truth")
    pinnLine, = ax.plot([], [], "o--", lw=2, color="#d62728", label="PINN")
    ax.legend(loc="upper right")

    def frame(i):
        trueLine.set_data(tx[i], ty[i])
        pinnLine.set_data(px[i], py[i])
        return trueLine, pinnLine

    anim = FuncAnimation(figAnim, frame, frames=steps, interval=WINDOW * 1000 / steps, blit=True)

    figPlot, axes = plt.subplots(2, 2, figsize=(11, 8))
    for k in range(3):
        a = axes[k // 2][k % 2]
        a.plot(tGrid, trueAngles[:, k], color="#1f77b4", label="truth")
        a.plot(tGrid, pinnAngles[:, k], "--", color="#d62728", label="PINN")
        a.set_title(f"theta{k + 1} vs time")
        a.set_xlabel("t (s)")
        a.legend()

    phase = axes[1][1]
    phase.plot(trueAngles[:, 0], trueAngles[:, 1], color="#1f77b4", label="truth")
    phase.plot(pinnAngles[:, 0], pinnAngles[:, 1], "--", color="#d62728", label="PINN")
    phase.set_title("phase: theta1 vs theta2")
    phase.set_xlabel("theta1")
    phase.set_ylabel("theta2")
    phase.legend()
    figPlot.tight_layout()
    plt.show()


mode = sys.argv[1] if len(sys.argv) > 1 else "run"
if mode == "train":
    train()
else:
    run(START_ANGLES)
