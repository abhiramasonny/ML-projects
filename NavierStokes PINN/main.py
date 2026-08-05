import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

here = os.path.dirname(os.path.abspath(__file__))
outDir = os.path.join(here, "outputs")
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

GHIA_Y = np.array([1.0, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0])
GHIA_X = np.array([1.0, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 0.5, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0])
GHIA_U = {
    100: np.array([1.0, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332, -0.13641, -0.20581, -0.2109, -0.15662, -0.1015, -0.06434, -0.04775, -0.04192, -0.03717, 0.0]),
    400: np.array([1.0, 0.75837, 0.68439, 0.61756, 0.55892, 0.29093, 0.16256, 0.02135, -0.11477, -0.17119, -0.32726, -0.24299, -0.14612, -0.10338, -0.09266, -0.08186, 0.0]),
    1000: np.array([1.0, 0.65928, 0.57492, 0.51117, 0.46604, 0.33304, 0.18719, 0.05702, -0.0608, -0.10648, -0.27805, -0.38289, -0.2973, -0.2222, -0.20196, -0.18109, 0.0]),
}
GHIA_V = {
    100: np.array([0.0, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914, -0.22445, -0.24533, 0.05454, 0.17527, 0.17507, 0.16077, 0.12317, 0.1089, 0.10091, 0.09233, 0.0]),
    400: np.array([0.0, -0.12146, -0.15663, -0.19254, -0.22847, -0.23827, -0.44993, -0.38598, 0.05186, 0.30174, 0.30203, 0.28124, 0.22965, 0.2092, 0.19713, 0.1836, 0.0]),
    1000: np.array([0.0, -0.21388, -0.27669, -0.33714, -0.39188, -0.5155, -0.42665, -0.31966, 0.02526, 0.32235, 0.33075, 0.37095, 0.32627, 0.30353, 0.29012, 0.27485, 0.0]),
}


def grad(out, inp):
    return torch.autograd.grad(out, inp, torch.ones_like(out), create_graph=True)[0]


def makeNet(nIn, nOut, width, depth):
    layers = [nn.Linear(nIn, width), nn.Tanh()]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), nn.Tanh()]
    layers += [nn.Linear(width, nOut)]
    net = nn.Sequential(*layers)
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
    return net.to(device)


def train(net, sample, loss, adamIters, lbfgsIters, lr):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(adamIters, 1))
    hist = []

    for it in range(adamIters):
        batch = sample()
        opt.zero_grad()
        value = loss(batch)
        value.backward()
        opt.step()
        sch.step()
        hist.append(value.item())
        if it % 500 == 0:
            print(f"adam {it}/{adamIters} loss {value.item():.3e}", flush=True)

    if lbfgsIters > 0:
        batch = sample()
        opt2 = torch.optim.LBFGS(net.parameters(), max_iter=lbfgsIters, history_size=50,
                                 tolerance_grad=1e-9, tolerance_change=1e-12, line_search_fn="strong_wolfe")

        def closure():
            opt2.zero_grad()
            value = loss(batch)
            value.backward()
            hist.append(value.item())
            return value

        print("lbfgs ...", flush=True)
        opt2.step(closure)
        print(f"lbfgs done loss {hist[-1]:.3e}", flush=True)

    return hist


def makeGrid(xlo, xhi, ylo, yhi, n):
    XX, YY = np.meshgrid(np.linspace(xlo, xhi, n), np.linspace(ylo, yhi, n))
    x = torch.tensor(XX.ravel()[:, None], dtype=torch.float32, device=device)
    y = torch.tensor(YY.ravel()[:, None], dtype=torch.float32, device=device)
    return XX, YY, x, y


def plotFields(path, title, XX, YY, items, stream=None):
    rows = (len(items) + 1) // 2
    fig, ax = plt.subplots(rows, 2, figsize=(12, 5 * rows), squeeze=False)
    for k, (name, Z, cmap) in enumerate(items):
        a = ax[k // 2][k % 2]
        c = a.contourf(XX, YY, Z, 60, cmap=cmap)
        if stream is not None and k == 0:
            a.streamplot(XX[0], YY[:, 0], stream[0], stream[1], color="white", density=1.3, linewidth=0.6, arrowsize=0.6)
        a.set_title(name)
        a.set_aspect("equal")
        a.set_xlabel("x")
        a.set_ylabel("y")
        fig.colorbar(c, ax=a)
    for k in range(len(items), rows * 2):
        ax[k // 2][k % 2].axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plotLoss(path, hist):
    fig, a = plt.subplots(figsize=(8, 4.5))
    a.semilogy(hist)
    a.set_xlabel("iteration")
    a.set_ylabel("loss")
    a.set_title("training loss")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def saveMetrics(name, metrics):
    with open(os.path.join(outDir, f"{name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2), flush=True)


def couette(adamIters=3000, lbfgsIters=800, lr=2e-3):
    nu = 0.1
    uTop = 1.0
    print(f"=== couette on {device} ===", flush=True)
    net = makeNet(2, 3, 48, 6)

    def flow(x, y):
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)
        out = net(torch.cat([2 * x - 1, 2 * y - 1], 1))
        u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]
        ux, uy = grad(u, x), grad(u, y)
        vx, vy = grad(v, x), grad(v, y)
        fu = u * ux + v * uy + grad(p, x) - nu * (grad(ux, x) + grad(uy, y))
        fv = u * vx + v * vy + grad(p, y) - nu * (grad(vx, x) + grad(vy, y))
        return u, v, p, fu, fv, ux + vy

    s = torch.rand(300, 1, device=device)
    z = torch.zeros(300, 1, device=device)
    o = torch.ones(300, 1, device=device)
    xWall = torch.cat([s, s])
    yWall = torch.cat([z, o])
    uWall = torch.cat([z, uTop + z])
    yEdge = torch.rand(300, 1, device=device)

    def sample():
        return torch.rand(4000, 1, device=device), torch.rand(4000, 1, device=device)

    def loss(batch):
        xi, yi = batch
        _, _, _, fu, fv, c = flow(xi, yi)
        uw, vw, _, _, _, _ = flow(xWall, yWall)
        ul, vl, pl, _, _, _ = flow(torch.zeros_like(yEdge), yEdge)
        ur, vr, pr, _, _, _ = flow(torch.ones_like(yEdge), yEdge)
        physics = (fu ** 2).mean() + (fv ** 2).mean() + (c ** 2).mean()
        walls = ((uw - uWall) ** 2).mean() + (vw ** 2).mean()
        periodic = ((ul - ur) ** 2).mean() + ((vl - vr) ** 2).mean() + ((pl - pr) ** 2).mean()
        return physics + 10 * walls + periodic

    hist = train(net, sample, loss, adamIters, lbfgsIters, lr)
    torch.save({"model": net.state_dict(), "hist": hist}, os.path.join(outDir, "couette.pth"))

    XX, YY, x, y = makeGrid(0, 1, 0, 1, 120)
    u, v, p, _, _, _ = flow(x, y)
    U = u.detach().cpu().numpy().reshape(120, 120)
    V = v.detach().cpu().numpy().reshape(120, 120)
    exact = uTop * YY

    plotFields(os.path.join(outDir, "couette_fields.png"), "Couette flow", XX, YY,
               [("u velocity + streamlines", U, "viridis"), ("abs u error", np.abs(U - exact), "magma"),
                ("v velocity", V, "coolwarm"), ("exact u", exact, "viridis")], stream=(U, V))
    plotLoss(os.path.join(outDir, "couette_loss.png"), hist)

    yLine = np.linspace(0, 1, 200)
    yt = torch.tensor(yLine[:, None], dtype=torch.float32, device=device)
    profile = flow(torch.full_like(yt, 0.5), yt)[0].detach().cpu().numpy().ravel()
    fig, a = plt.subplots(figsize=(6, 5))
    a.plot(profile, yLine, "-", color="#d62728", label="PINN")
    a.plot(uTop * yLine, yLine, "o", ms=3, color="#1f77b4", label="exact linear")
    a.set_xlabel("u")
    a.set_ylabel("y")
    a.set_title("u profile at x=0.5")
    a.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outDir, "couette_profile.png"), dpi=130)
    plt.close(fig)

    saveMetrics("couette", {"velocity_rel_l2": float(np.linalg.norm(U - exact) / np.linalg.norm(exact)),
                            "final_loss": float(hist[-1])})


def poiseuille(adamIters=3000, lbfgsIters=800, lr=2e-3):
    nu = 0.1
    pIn = 0.8
    print(f"=== poiseuille on {device} ===", flush=True)
    net = makeNet(2, 3, 48, 6)

    def flow(x, y):
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)
        out = net(torch.cat([2 * x - 1, 2 * y - 1], 1))
        u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]
        ux, uy = grad(u, x), grad(u, y)
        vx, vy = grad(v, x), grad(v, y)
        fu = u * ux + v * uy + grad(p, x) - nu * (grad(ux, x) + grad(uy, y))
        fv = u * vx + v * vy + grad(p, y) - nu * (grad(vx, x) + grad(vy, y))
        return u, v, p, fu, fv, ux + vy

    s = torch.rand(300, 1, device=device)
    z = torch.zeros(300, 1, device=device)
    o = torch.ones(300, 1, device=device)
    xWall = torch.cat([s, s])
    yWall = torch.cat([z, o])
    xEnds = torch.cat([z, o])
    yEnds = torch.cat([s, s])
    pEnds = torch.cat([pIn + z, z])

    def sample():
        return torch.rand(4000, 1, device=device), torch.rand(4000, 1, device=device)

    def loss(batch):
        xi, yi = batch
        _, _, _, fu, fv, c = flow(xi, yi)
        uw, vw, _, _, _, _ = flow(xWall, yWall)
        _, ve, pe, _, _, _ = flow(xEnds, yEnds)
        physics = (fu ** 2).mean() + (fv ** 2).mean() + (c ** 2).mean()
        walls = (uw ** 2).mean() + (vw ** 2).mean()
        ends = ((pe - pEnds) ** 2).mean() + (ve ** 2).mean()
        return physics + 10 * (walls + ends)

    hist = train(net, sample, loss, adamIters, lbfgsIters, lr)
    torch.save({"model": net.state_dict(), "hist": hist}, os.path.join(outDir, "poiseuille.pth"))

    XX, YY, x, y = makeGrid(0, 1, 0, 1, 120)
    u, v, p, _, _, _ = flow(x, y)
    U = u.detach().cpu().numpy().reshape(120, 120)
    V = v.detach().cpu().numpy().reshape(120, 120)
    P = p.detach().cpu().numpy().reshape(120, 120)
    exact = pIn / (2 * nu) * YY * (1 - YY)

    plotFields(os.path.join(outDir, "poiseuille_fields.png"), "Poiseuille channel flow", XX, YY,
               [("u velocity + streamlines", U, "viridis"), ("pressure", P, "coolwarm"),
                ("abs u error", np.abs(U - exact), "magma"), ("v velocity", V, "coolwarm")], stream=(U, V))
    plotLoss(os.path.join(outDir, "poiseuille_loss.png"), hist)

    yLine = np.linspace(0, 1, 200)
    yt = torch.tensor(yLine[:, None], dtype=torch.float32, device=device)
    profile = flow(torch.full_like(yt, 0.5), yt)[0].detach().cpu().numpy().ravel()
    fig, a = plt.subplots(figsize=(6, 5))
    a.plot(profile, yLine, "-", color="#d62728", label="PINN")
    a.plot(pIn / (2 * nu) * yLine * (1 - yLine), yLine, "o", ms=3, color="#1f77b4", label="exact parabola")
    a.set_xlabel("u")
    a.set_ylabel("y")
    a.set_title("u profile at x=0.5")
    a.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outDir, "poiseuille_profile.png"), dpi=130)
    plt.close(fig)

    saveMetrics("poiseuille", {"velocity_rel_l2": float(np.linalg.norm(U - exact) / np.linalg.norm(exact)),
                               "final_loss": float(hist[-1])})


def kovasznay(adamIters=4000, lbfgsIters=1500, lr=2e-3):
    re = 40.0
    nu = 1.0 / re
    lam = re / 2 - np.sqrt(re ** 2 / 4 + 4 * np.pi ** 2)
    xlo, xhi, ylo, yhi = -0.5, 1.0, -0.5, 1.5
    print(f"=== kovasznay Re={int(re)} on {device} ===", flush=True)
    net = makeNet(2, 3, 64, 8)

    def exact(x, y):
        u = 1 - np.exp(lam * x) * np.cos(2 * np.pi * y)
        v = lam / (2 * np.pi) * np.exp(lam * x) * np.sin(2 * np.pi * y)
        p = 0.5 * (1 - np.exp(2 * lam * x))
        return u, v, p

    def flow(x, y):
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)
        out = net(torch.cat([2 * (x - xlo) / (xhi - xlo) - 1, 2 * (y - ylo) / (yhi - ylo) - 1], 1))
        u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]
        ux, uy = grad(u, x), grad(u, y)
        vx, vy = grad(v, x), grad(v, y)
        fu = u * ux + v * uy + grad(p, x) - nu * (grad(ux, x) + grad(uy, y))
        fv = u * vx + v * vy + grad(p, y) - nu * (grad(vx, x) + grad(vy, y))
        return u, v, p, fu, fv, ux + vy

    t = torch.rand(400, 1)
    z = torch.zeros(400, 1)
    xBnd = torch.cat([xlo + z, xhi + z, xlo + t * (xhi - xlo), xlo + t * (xhi - xlo)]).to(device)
    yBnd = torch.cat([ylo + t * (yhi - ylo), ylo + t * (yhi - ylo), ylo + z, yhi + z]).to(device)
    ue, ve, pe = exact(xBnd.cpu().numpy(), yBnd.cpu().numpy())
    uBnd = torch.tensor(ue, dtype=torch.float32, device=device)
    vBnd = torch.tensor(ve, dtype=torch.float32, device=device)
    pBnd = torch.tensor(pe, dtype=torch.float32, device=device)

    def sample():
        x = torch.rand(4000, 1, device=device) * (xhi - xlo) + xlo
        y = torch.rand(4000, 1, device=device) * (yhi - ylo) + ylo
        return x, y

    def loss(batch):
        xi, yi = batch
        _, _, _, fu, fv, c = flow(xi, yi)
        up, vp, pp, _, _, _ = flow(xBnd, yBnd)
        physics = (fu ** 2).mean() + (fv ** 2).mean() + (c ** 2).mean()
        bnd = ((up - uBnd) ** 2).mean() + ((vp - vBnd) ** 2).mean() + ((pp - pBnd) ** 2).mean()
        return physics + 10 * bnd

    hist = train(net, sample, loss, adamIters, lbfgsIters, lr)
    torch.save({"model": net.state_dict(), "hist": hist}, os.path.join(outDir, "kovasznay.pth"))

    XX, YY, x, y = makeGrid(xlo, xhi, ylo, yhi, 120)
    u, v, p, _, _, _ = flow(x, y)
    U = u.detach().cpu().numpy().reshape(120, 120)
    V = v.detach().cpu().numpy().reshape(120, 120)
    ueGrid, veGrid, _ = exact(XX, YY)
    err = np.sqrt((U - ueGrid) ** 2 + (V - veGrid) ** 2)

    plotFields(os.path.join(outDir, "kovasznay_fields.png"), f"Kovasznay flow  Re={int(re)}", XX, YY,
               [("PINN speed + streamlines", np.sqrt(U ** 2 + V ** 2), "viridis"),
                ("exact speed", np.sqrt(ueGrid ** 2 + veGrid ** 2), "viridis"),
                ("abs velocity error", err, "magma"), ("PINN u", U, "coolwarm")], stream=(U, V))
    plotLoss(os.path.join(outDir, "kovasznay_loss.png"), hist)

    rel = np.linalg.norm(np.stack([U - ueGrid, V - veGrid])) / np.linalg.norm(np.stack([ueGrid, veGrid]))
    saveMetrics("kovasznay", {"re": int(re), "velocity_rel_l2": float(rel),
                              "max_abs_err": float(err.max()), "final_loss": float(hist[-1])})


def taylorgreen(adamIters=6000, lbfgsIters=1500, lr=2e-3):
    nu = 0.1
    tEnd = 3.0
    twoPi = 2 * np.pi
    print(f"=== taylor-green nu={nu} on {device} ===", flush=True)
    net = makeNet(3, 3, 64, 8)

    def flow(x, y, t):
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)
        t = t.requires_grad_(True)
        out = net(torch.cat([x / np.pi - 1, y / np.pi - 1, 2 * t / tEnd - 1], 1))
        u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]
        ux, uy = grad(u, x), grad(u, y)
        vx, vy = grad(v, x), grad(v, y)
        fu = grad(u, t) + u * ux + v * uy + grad(p, x) - nu * (grad(ux, x) + grad(uy, y))
        fv = grad(v, t) + u * vx + v * vy + grad(p, y) - nu * (grad(vx, x) + grad(vy, y))
        return u, v, p, fu, fv, ux + vy

    x0 = torch.rand(2000, 1, device=device) * twoPi
    y0 = torch.rand(2000, 1, device=device) * twoPi
    u0 = torch.sin(x0) * torch.cos(y0)
    v0 = -torch.cos(x0) * torch.sin(y0)

    tb = torch.rand(1500, 1, device=device) * tEnd
    sb = torch.rand(1500, 1, device=device) * twoPi
    zb = torch.zeros(1500, 1, device=device)
    pb = torch.full((1500, 1), twoPi, device=device)
    xBnd = torch.cat([zb, pb, sb, sb])
    yBnd = torch.cat([sb, sb, zb, pb])
    tBnd = torch.cat([tb, tb, tb, tb])
    uBnd = torch.sin(xBnd) * torch.cos(yBnd) * torch.exp(-2 * nu * tBnd)
    vBnd = -torch.cos(xBnd) * torch.sin(yBnd) * torch.exp(-2 * nu * tBnd)

    def sample():
        return (torch.rand(5000, 1, device=device) * twoPi,
                torch.rand(5000, 1, device=device) * twoPi,
                torch.rand(5000, 1, device=device) * tEnd)

    def loss(batch):
        x, y, t = batch
        _, _, _, fu, fv, c = flow(x, y, t)
        up, vp, _, _, _, _ = flow(x0, y0, torch.zeros_like(x0))
        ub, vb, _, _, _, _ = flow(xBnd, yBnd, tBnd)
        physics = (fu ** 2).mean() + (fv ** 2).mean() + (c ** 2).mean()
        initial = ((up - u0) ** 2).mean() + ((vp - v0) ** 2).mean()
        bnd = ((ub - uBnd) ** 2).mean() + ((vb - vBnd) ** 2).mean()
        return physics + 10 * initial + 10 * bnd

    hist = train(net, sample, loss, adamIters, lbfgsIters, lr)
    torch.save({"model": net.state_dict(), "hist": hist}, os.path.join(outDir, "taylorgreen.pth"))
    plotLoss(os.path.join(outDir, "taylorgreen_loss.png"), hist)

    n = 80
    xs = np.linspace(0, twoPi, n)
    XX, YY = np.meshgrid(xs, xs)
    xf = torch.tensor(XX.ravel()[:, None], dtype=torch.float32, device=device)
    yf = torch.tensor(YY.ravel()[:, None], dtype=torch.float32, device=device)
    times = np.linspace(0, tEnd, 60)
    preds, exacts, errs = [], [], []

    for tv in times:
        u, v, _, _, _, _ = flow(xf, yf, torch.full_like(xf, float(tv)))
        U = u.detach().cpu().numpy().reshape(n, n)
        V = v.detach().cpu().numpy().reshape(n, n)
        decay = np.exp(-2 * nu * tv)
        ue = np.sin(XX) * np.cos(YY) * decay
        ve = -np.cos(XX) * np.sin(YY) * decay
        preds.append(np.sqrt(U ** 2 + V ** 2))
        exacts.append(np.sqrt(ue ** 2 + ve ** 2))
        errs.append(float(np.linalg.norm(np.stack([U - ue, V - ve])) / (np.linalg.norm(np.stack([ue, ve])) + 1e-9)))

    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    im0 = ax[0].imshow(preds[0], origin="lower", extent=[0, twoPi, 0, twoPi], cmap="viridis", vmin=0, vmax=1)
    im1 = ax[1].imshow(exacts[0], origin="lower", extent=[0, twoPi, 0, twoPi], cmap="viridis", vmin=0, vmax=1)
    ax[0].set_title("PINN speed")
    ax[1].set_title("exact speed")
    title = fig.suptitle("t=0.00")
    fig.colorbar(im0, ax=ax[0])
    fig.colorbar(im1, ax=ax[1])

    def frame(i):
        im0.set_data(preds[i])
        im1.set_data(exacts[i])
        title.set_text(f"t={times[i]:.2f}")
        return im0, im1, title

    anim = FuncAnimation(fig, frame, frames=len(times), interval=80, blit=False)
    anim.save(os.path.join(outDir, "taylorgreen.gif"), writer=PillowWriter(fps=12))
    plt.close(fig)

    fig, a = plt.subplots(figsize=(8, 4.5))
    a.plot(times, errs, "-o", ms=3, color="#d62728")
    a.set_xlabel("t")
    a.set_ylabel("velocity rel L2 error")
    a.set_title("Taylor-Green error vs time")
    fig.tight_layout()
    fig.savefig(os.path.join(outDir, "taylorgreen_error.png"), dpi=130)
    plt.close(fig)

    saveMetrics("taylorgreen", {"nu": nu, "T": tEnd, "mean_rel_l2": float(np.mean(errs)),
                                "max_rel_l2": float(np.max(errs)), "final_loss": float(hist[-1])})


def cavity(reList=(100, 400, 1000), adamIters=6000, lbfgsIters=1500, lr=2e-3, width=64, depth=8):
    results = []
    warmStart = None

    for i, re in enumerate(sorted(reList)):
        nu = 1.0 / re
        iters = adamIters if i == 0 else max(adamIters // 2, 2500)
        print(f"=== cavity Re={re} on {device} ===", flush=True)
        net = makeNet(2, 2, width, depth)
        if warmStart is not None:
            net.load_state_dict(warmStart)

        def flow(x, y):
            x = x.requires_grad_(True)
            y = y.requires_grad_(True)
            out = net(torch.cat([2 * x - 1, 2 * y - 1], 1))
            psi, p = out[:, 0:1], out[:, 1:2]
            u = grad(psi, y)
            v = -grad(psi, x)
            ux, uy = grad(u, x), grad(u, y)
            vx, vy = grad(v, x), grad(v, y)
            fu = u * ux + v * uy + grad(p, x) - nu * (grad(ux, x) + grad(uy, y))
            fv = u * vx + v * vy + grad(p, y) - nu * (grad(vx, x) + grad(vy, y))
            return u, v, p, fu, fv, vx - uy

        t = torch.linspace(0, 1, 200)
        z, o = torch.zeros(200), torch.ones(200)
        xBnd = torch.cat([t, t, z, o]).reshape(-1, 1).to(device)
        yBnd = torch.cat([z, o, t, t]).reshape(-1, 1).to(device)
        uBnd = torch.cat([z, o, z, z]).reshape(-1, 1).to(device)
        vBnd = torch.zeros(800, 1, device=device)

        def sample():
            xi = torch.rand(4000, 1, device=device)
            yi = torch.rand(4000, 1, device=device)
            r = lambda: torch.rand(400, 1, device=device)
            xw = torch.cat([r(), r(), r() * 0.06, 1 - r() * 0.06])
            yw = torch.cat([r() * 0.06, 1 - r() * 0.06, r(), r()])
            return torch.cat([xi, xw]), torch.cat([yi, yw])

        def loss(batch):
            xi, yi = batch
            _, _, _, fu, fv, _ = flow(xi, yi)
            up, vp, _, _, _, _ = flow(xBnd, yBnd)
            return (fu ** 2).mean() + (fv ** 2).mean() + 10 * (((up - uBnd) ** 2).mean() + ((vp - vBnd) ** 2).mean())

        hist = train(net, sample, loss, iters, lbfgsIters, lr)
        torch.save({"model": net.state_dict(), "re": re, "hist": hist},
                   os.path.join(outDir, f"cavity_re{re}.pth"))
        warmStart = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

        XX, YY, x, y = makeGrid(0, 1, 0, 1, 120)
        u, v, p, _, _, w = flow(x, y)
        U = u.detach().cpu().numpy().reshape(120, 120)
        V = v.detach().cpu().numpy().reshape(120, 120)
        P = p.detach().cpu().numpy().reshape(120, 120)
        W = w.detach().cpu().numpy().reshape(120, 120)

        plotFields(os.path.join(outDir, f"cavity_re{re}_fields.png"), f"Lid-driven cavity  Re={re}", XX, YY,
                   [("velocity magnitude + streamlines", np.sqrt(U ** 2 + V ** 2), "viridis"),
                    ("pressure", P, "coolwarm"), ("vorticity", W, "RdBu_r"), ("u velocity", U, "viridis")],
                   stream=(U, V))
        plotLoss(os.path.join(outDir, f"cavity_re{re}_loss.png"), hist)

        yl = torch.linspace(0, 1, 200, device=device).reshape(-1, 1)
        xh = torch.linspace(0, 1, 200, device=device).reshape(-1, 1)
        uCenter = flow(torch.full_like(yl, 0.5), yl)[0].detach().cpu().numpy().ravel()
        vCenter = flow(xh, torch.full_like(xh, 0.5))[1].detach().cpu().numpy().ravel()
        yLine = yl.detach().cpu().numpy().ravel()
        xLine = xh.detach().cpu().numpy().ravel()

        results.append({
            "re": re,
            "u_rel_l2": float(np.linalg.norm(np.interp(GHIA_Y, yLine, uCenter) - GHIA_U[re]) / np.linalg.norm(GHIA_U[re])),
            "v_rel_l2": float(np.linalg.norm(np.interp(GHIA_X, xLine, vCenter) - GHIA_V[re]) / np.linalg.norm(GHIA_V[re])),
            "final_loss": float(hist[-1]),
            "lines": (yLine, uCenter, xLine, vCenter),
        })

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = {100: "#1f77b4", 400: "#2ca02c", 1000: "#d62728"}
    for r in results:
        yLine, uCenter, xLine, vCenter = r["lines"]
        c = colors.get(r["re"], "#333333")
        ax[0].plot(uCenter, yLine, "-", color=c, label=f"PINN Re={r['re']}")
        ax[0].plot(GHIA_U[r["re"]], GHIA_Y, "o", color=c, ms=4)
        ax[1].plot(xLine, vCenter, "-", color=c, label=f"PINN Re={r['re']}")
        ax[1].plot(GHIA_X, GHIA_V[r["re"]], "o", color=c, ms=4)
    ax[0].set_xlabel("u")
    ax[0].set_ylabel("y")
    ax[0].set_title("u along x=0.5 (dots = Ghia 1982)")
    ax[0].legend()
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("v")
    ax[1].set_title("v along y=0.5 (dots = Ghia 1982)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outDir, "cavity_centerlines.png"), dpi=130)
    plt.close(fig)

    saveMetrics("cavity", [{k: r[k] for k in ("re", "u_rel_l2", "v_rel_l2", "final_loss")} for r in results])


cases = {"couette": couette, "poiseuille": poiseuille, "kovasznay": kovasznay,
         "taylorgreen": taylorgreen, "cavity": cavity}

os.makedirs(outDir, exist_ok=True)
name = sys.argv[1] if len(sys.argv) > 1 else "cavity"
if name not in cases:
    sys.exit(f"pick one of: {', '.join(cases)}")
cases[name]()
