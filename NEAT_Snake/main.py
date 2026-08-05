import os
import sys
import pickle
import multiprocessing
import tkinter as tk
import neat
from neat import Population, IN, HID, OUT, BIAS
from snake import Snake, chooseAction, GRID, NIN

here = os.path.dirname(os.path.abspath(__file__))
statePath = os.path.join(here, "pop.pkl")

GENERATIONS = 200
PLAY_SPEED = 150

RAY_NAMES = ["S", "FR", "R", "BR", "B", "BL", "L", "FL"]
IN_LABELS = ([RAY_NAMES[i // 3] + "wbf"[i % 3] for i in range(24)]
             + ["fF", "fR", "ln", "reach"]
             + ["dL", "sL", "dS", "sS", "dR", "sR"]
             + ["freeL", "freeS", "freeR"]
             + ["safeL", "safeS", "safeR"]
             + ["tailF", "tailR", "tailD"])
OUT_LABELS = ["<L", "^S", "R>"]


def saveState(pop):
    with open(statePath, "wb") as f:
        pickle.dump({"nin": NIN, "grid": GRID, "pop": pop, "innovDb": dict(neat.innovDb),
                     "innovCount": neat.innovCount[0], "nodeDb": dict(neat.nodeDb),
                     "nextNode": neat.nextNode[0]}, f)


def loadState():
    if not os.path.exists(statePath):
        return Population()
    try:
        with open(statePath, "rb") as f:
            payload = pickle.load(f)
        if payload.get("nin") != NIN or payload.get("grid") != GRID:
            print("incompatible state ignored; starting fresh")
            return Population()
        neat.innovDb.update(payload["innovDb"])
        neat.innovCount[0] = payload["innovCount"]
        neat.nodeDb.update(payload["nodeDb"])
        neat.nextNode[0] = payload["nextNode"]
        print("resumed from gen", payload["pop"].gen)
        return payload["pop"]
    except Exception as e:
        print("could not load state:", e, "- starting fresh")
        return Population()


def drawBrain(axNet, axFit, g, gen, hist):
    ins = [i for i, nd in g.nodes.items() if nd.kind in (IN, BIAS)]
    depth = {i: 0 for i in ins}
    for _ in range(len(g.nodes)):
        changed = False
        for c in g.conns.values():
            if not c.on or c.s not in depth:
                continue
            if c.d not in depth or depth[c.d] < depth[c.s] + 1:
                depth[c.d] = depth[c.s] + 1
                changed = True
        if not changed:
            break

    maxDepth = max(depth.values()) if depth else 0
    for i, nd in g.nodes.items():
        if nd.kind == OUT:
            depth[i] = maxDepth + 1
        elif i not in depth:
            depth[i] = max(1, maxDepth // 2)
    maxDepth = max(depth.values()) if depth else 1

    byDepth = {}
    for i, d in depth.items():
        byDepth.setdefault(d, []).append(i)
    pos = {}
    for d, ns in byDepth.items():
        for j, i in enumerate(sorted(ns)):
            pos[i] = (d / maxDepth, (j + 1) / (len(ns) + 1))

    axNet.clear()
    axNet.set_xlim(-0.08, 1.08)
    axNet.set_ylim(0, 1)
    axNet.axis("off")
    axNet.set_facecolor("#000000")

    for c in g.conns.values():
        if not c.on or c.s not in pos or c.d not in pos:
            continue
        x0, y0 = pos[c.s]
        x1, y1 = pos[c.d]
        axNet.plot([x0, x1], [y0, y1], color="#ffffff" if c.w > 0 else "#444444",
                   alpha=min(0.7, 0.1 + abs(c.w) * 0.25), lw=min(1.5, 0.3 + abs(c.w) * 0.4), zorder=1)

    for i, nd in g.nodes.items():
        if i not in pos:
            continue
        x, y = pos[i]
        color = {OUT: "#ffffff", BIAS: "#222222", HID: "#888888", IN: "#ffffff"}[nd.kind]
        axNet.scatter(x, y, s=120 if nd.kind in (IN, BIAS) else 180, color=color,
                      zorder=3, edgecolors="#ffffff", linewidths=0.5)
        if i < NIN:
            label = IN_LABELS[i]
        elif nd.kind == OUT:
            label = OUT_LABELS[i - NIN - 1]
        elif nd.kind == BIAS:
            label = "B"
        else:
            label = ""
        if label:
            axNet.text(x, y, label, ha="center", va="center", fontsize=4,
                       color="#000000" if nd.kind == OUT else "#ffffff", zorder=4)

    axNet.set_title(f"gen {gen}   fit {g.fit:.0f}   n{len(g.nodes)} c{len(g.conns)}",
                    color="#888888", fontsize=8, pad=8, fontfamily="monospace")

    axFit.clear()
    axFit.set_facecolor("#000000")
    for sp in axFit.spines.values():
        sp.set_color("#222222")
    axFit.tick_params(colors="#444444", labelsize=7)
    axFit.set_xlabel("gen", color="#444444", fontsize=7, fontfamily="monospace")
    axFit.set_ylabel("fitness", color="#444444", fontsize=7, fontfamily="monospace")
    if hist:
        axFit.plot(hist, color="#ffffff", lw=1.0, alpha=0.9)
        axFit.fill_between(range(len(hist)), hist, alpha=0.06, color="#ffffff")


def train(gens=GENERATIONS, viz=False, reset=False):
    if reset and os.path.exists(statePath):
        os.remove(statePath)

    pop = loadState()
    pool = multiprocessing.get_context("fork").Pool(maxtasksperchild=250)
    print("workers:", pool._processes, "| episodes:", neat.EPISODES)

    hist = []
    fig = axNet = axFit = None
    if viz:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        plt.ion()
        fig, (axNet, axFit) = plt.subplots(1, 2, figsize=(13, 6), facecolor="#000000")
        fig.subplots_adjust(left=0.05, right=0.97, bottom=0.1, top=0.93, wspace=0.3)
        plt.show(block=False)

    try:
        for i in range(gens):
            best = pop.step(pool)
            hist.append(best.fit)
            print(f"gen {pop.gen:4d} | fit {best.fit:9.1f} | ever {pop.best.fit:9.1f} | "
                  f"sp {len(pop.species):3d} | compat {neat.compat:.2f} | "
                  f"n{len(best.nodes)} c{len(best.conns)}")
            if viz:
                drawBrain(axNet, axFit, best, pop.gen, hist)
                fig.canvas.draw()
                fig.canvas.flush_events()
            if i % 10 == 0 or i == gens - 1:
                saveState(pop)
    except KeyboardInterrupt:
        print("\ninterrupted; saving state")
        saveState(pop)
    finally:
        pool.close()
        pool.join()

    if pop.best:
        print("done. best:", pop.best.fit)


def play(speed=PLAY_SPEED):
    if not os.path.exists(statePath):
        print("no saved genome, run: uv run main.py train")
        return

    with open(statePath, "rb") as f:
        pop = pickle.load(f)["pop"]
    genome = pop.best or max(pop.genomes, key=lambda g: g.fit)
    print(f"playing gen {pop.gen} best, fitness {genome.fit:.0f}")

    cell = 36
    size = GRID * cell
    root = tk.Tk()
    root.title("NEAT Snake")
    root.configure(bg="#000000")
    cv = tk.Canvas(root, width=size + 160, height=size, bg="#000000", highlightthickness=0)
    cv.pack()

    s = Snake()
    live = [True]

    def draw():
        cv.delete("all")
        for x in range(GRID):
            for y in range(GRID):
                cv.create_rectangle(x * cell, y * cell, x * cell + cell, y * cell + cell,
                                    fill="#0a0a0a", outline="#111111", width=1)
        if s.food:
            fx, fy = s.food
            cv.create_rectangle(fx * cell + 2, fy * cell + 2, fx * cell + cell - 2, fy * cell + cell - 2,
                                fill="#cc2222", outline="")
        for i, (bx, by) in enumerate(s.body):
            cv.create_rectangle(bx * cell + 2, by * cell + 2, bx * cell + cell - 2, by * cell + cell - 2,
                                fill="#44cc44" if i == 0 else "#226622", outline="")

        x0 = size + 18
        mono = ("Courier", 10)
        cv.create_text(x0, 20, text="NEAT SNAKE", fill="#ffffff", font=("Courier", 11, "bold"), anchor="w")
        cv.create_text(x0, 52, text=f"apples  {s.apples:>4}", fill="#ffffff", font=mono, anchor="w")
        cv.create_text(x0, 72, text=f"steps   {s.steps:>4}", fill="#444444", font=mono, anchor="w")
        cv.create_text(x0, 92, text=f"nodes   {len(genome.nodes):>4}", fill="#444444", font=mono, anchor="w")
        cv.create_text(x0, 112, text=f"conns   {len(genome.conns):>4}", fill="#444444", font=mono, anchor="w")
        cv.create_text(x0, size - 16, text="alive" if live[0] else "dead",
                       fill="#ffffff" if live[0] else "#555555", font=mono, anchor="w")

    def tick():
        if live[0]:
            live[0] = s.step(chooseAction(genome, s))
        else:
            s.reset()
            live[0] = True
        draw()
        root.after(speed, tick)

    tick()
    root.mainloop()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    if mode == "play":
        play()
    else:
        train(viz="--viz" in sys.argv, reset="--reset" in sys.argv)
