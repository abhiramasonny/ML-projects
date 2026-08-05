import math
import random
from collections import deque
from snake import Snake, chooseAction, NIN, NOUT

POP_SIZE = 400
TARGET_SPECIES = 18
COMPAT_STEP = 0.03
C1, C3 = 1.0, 0.4
W_MUTATE, W_PERTURB = 0.85, 0.35
ADD_CONN, ADD_NODE = 0.04, 0.02
SURVIVAL_RATE, CROSSOVER_RATE = 0.25, 0.75
STAGNATION = 40
EPISODES = 8

compat = 1.5
seedBank = [random.Random(12345).randrange(10 ** 9) for _ in range(256)]

innovDb = {}
innovCount = [0]
nodeDb = {}
nextNode = [NIN + 1 + NOUT]

IN, HID, OUT, BIAS = 0, 1, 2, 3


def innov(s, d):
    if (s, d) not in innovDb:
        innovDb[(s, d)] = innovCount[0]
        innovCount[0] += 1
    return innovDb[(s, d)]


def splitNodeId(connInnov):
    if connInnov not in nodeDb:
        nodeDb[connInnov] = nextNode[0]
        nextNode[0] += 1
    return nodeDb[connInnov]


class Node:
    __slots__ = ("nid", "kind")

    def __init__(self, nid, kind):
        self.nid = nid
        self.kind = kind


class Conn:
    __slots__ = ("s", "d", "w", "on", "inv")

    def __init__(self, s, d, w, on=True):
        self.s = s
        self.d = d
        self.w = w
        self.on = on
        self.inv = innov(s, d)


class Genome:
    def __init__(self):
        self.nodes = {}
        self.conns = {}
        self.fit = 0.0
        self.adj = 0.0
        self.order = None
        self.incoming = None

    def __getstate__(self):
        return dict(nodes=self.nodes, conns=self.conns, fit=self.fit, adj=self.adj)

    def __setstate__(self, state):
        self.nodes = state.get("nodes", {})
        self.conns = state.get("conns", {})
        self.fit = state.get("fit", 0.0)
        self.adj = state.get("adj", 0.0)
        self.order = None
        self.incoming = None

    def compile(self):
        if self.order is not None:
            return

        adj = {n: [] for n in self.nodes}
        deg = {i: 0 for i, nd in self.nodes.items() if nd.kind not in (IN, BIAS)}
        self.incoming = {i: [] for i in self.nodes}

        for c in self.conns.values():
            if not c.on:
                continue
            if c.s in adj:
                adj[c.s].append(c.d)
            if c.d in deg:
                deg[c.d] += 1
            self.incoming.setdefault(c.d, []).append(c)

        queue = deque(i for i in deg if deg[i] == 0)
        order = []
        while queue:
            n = queue.popleft()
            order.append(n)
            for d in adj.get(n, []):
                if d in deg:
                    deg[d] -= 1
                    if deg[d] == 0:
                        queue.append(d)

        for n in deg:
            if n not in order:
                order.append(n)
        self.order = order

    def activate(self, x):
        self.compile()
        v = {i: float(x[i]) for i in range(NIN)}
        v[NIN] = 1.0
        for i, nd in self.nodes.items():
            if nd.kind not in (IN, BIAS):
                v[i] = 0.0
        for i in self.order:
            total = 0.0
            for c in self.incoming.get(i, []):
                total += c.w * v.get(c.s, 0.0)
            v[i] = math.tanh(total)
        return [v.get(NIN + 1 + i, 0.0) for i in range(NOUT)]

    def mutate(self):
        changed = False

        for c in self.conns.values():
            if random.random() >= W_MUTATE:
                continue
            if random.random() < 0.9:
                c.w += random.gauss(0.0, W_PERTURB)
            else:
                c.w = random.uniform(-2.0, 2.0)
            c.w = max(-4.0, min(4.0, c.w))

        if random.random() < ADD_CONN:
            srcs = [i for i, nd in self.nodes.items() if nd.kind != OUT]
            dsts = [i for i, nd in self.nodes.items() if nd.kind not in (IN, BIAS)]
            for _ in range(30):
                if not srcs or not dsts:
                    break
                s = random.choice(srcs)
                d = random.choice(dsts)
                if s == d or createsCycle(self, s, d):
                    continue
                inv = innov(s, d)
                if inv not in self.conns:
                    self.conns[inv] = Conn(s, d, random.uniform(-2.0, 2.0))
                    changed = True
                    break

        if random.random() < ADD_NODE:
            active = [c for c in self.conns.values() if c.on]
            if active:
                c = random.choice(active)
                c.on = False
                changed = True
                nid = splitNodeId(c.inv)
                if nid not in self.nodes:
                    self.nodes[nid] = Node(nid, HID)
                self.conns[innov(c.s, nid)] = Conn(c.s, nid, 1.0)
                self.conns[innov(nid, c.d)] = Conn(nid, c.d, c.w)

        if changed:
            self.order = None
            self.incoming = None

    def dist(self, other):
        k1 = set(self.conns)
        k2 = set(other.conns)
        matching = k1 & k2
        n = max(len(k1), len(k2), 1)
        if matching:
            wd = sum(abs(self.conns[k].w - other.conns[k].w) for k in matching) / len(matching)
        else:
            wd = 0.0
        return C1 * len(k1 ^ k2) / n + C3 * wd

    def copy(self):
        g = Genome()
        g.nodes = {i: Node(i, nd.kind) for i, nd in self.nodes.items()}
        g.conns = {inv: Conn(c.s, c.d, c.w, c.on) for inv, c in self.conns.items()}
        g.fit = self.fit
        g.adj = self.adj
        return g


def createsCycle(g, s, d):
    stack = [d]
    seen = set()
    while stack:
        n = stack.pop()
        if n == s:
            return True
        if n in seen:
            continue
        seen.add(n)
        for c in g.conns.values():
            if c.on and c.s == n:
                stack.append(c.d)
    return False


def cross(p1, p2):
    if p2.fit > p1.fit:
        p1, p2 = p2, p1

    child = Genome()
    child.nodes = {i: Node(i, nd.kind) for i, nd in p1.nodes.items()}

    for inv in set(p1.conns) | set(p2.conns):
        if inv in p1.conns and inv in p2.conns:
            source = random.choice([p1.conns[inv], p2.conns[inv]])
        elif inv in p1.conns:
            source = p1.conns[inv]
        else:
            continue
        if source.s not in child.nodes or source.d not in child.nodes:
            continue
        enabled = source.on
        if inv in p1.conns and inv in p2.conns:
            if (not p1.conns[inv].on or not p2.conns[inv].on) and random.random() < 0.75:
                enabled = False
        child.conns[inv] = Conn(source.s, source.d, source.w, enabled)

    return child


def seedGenome():
    g = Genome()
    for i in range(NIN):
        g.nodes[i] = Node(i, IN)
    g.nodes[NIN] = Node(NIN, BIAS)
    for i in range(NOUT):
        g.nodes[NIN + 1 + i] = Node(NIN + 1 + i, OUT)
    for s in range(NIN + 1):
        for i in range(NOUT):
            if random.random() < 0.65:
                d = NIN + 1 + i
                g.conns[innov(s, d)] = Conn(s, d, random.uniform(-1.0, 1.0))
    return g


def evalGenome(args):
    g, seeds = args
    total = 0.0

    for sd in seeds:
        s = Snake(rng=random.Random(sd))
        r = 0.0
        hx, hy = s.body[0]
        gx, gy = s.food if s.food is not None else (hx, hy)
        prevDist = abs(hx - gx) + abs(hy - gy)
        sinceApple = 0
        uniqueCells = {s.body[0]}
        alive = True

        while alive:
            prevApples = s.apples
            alive = s.step(chooseAction(g, s))
            sinceApple += 1

            if not alive:
                r -= 500.0 + 80.0 * len(s.body) + 250.0 * s.apples
                break

            hx, hy = s.body[0]
            uniqueCells.add((hx, hy))

            if sinceApple > 100 + 10 * len(s.body):
                r -= 250.0 + 30.0 * len(s.body)
                break

            if s.apples > prevApples:
                tailOk, free, margin = s.survivalScore(s.body)
                if tailOk and margin >= 0:
                    r += 700.0 + 180.0 * s.apples + 20.0 * margin
                else:
                    r -= 900.0 + 200.0 * s.apples
                if s.food is not None:
                    gx, gy = s.food
                    prevDist = abs(hx - gx) + abs(hy - gy)
                sinceApple = 0
            else:
                d = abs(hx - gx) + abs(hy - gy)
                foodIsSafe = max(s.safeFoodAfter(0), s.safeFoodAfter(1), s.safeFoodAfter(2)) > 0.0
                if foodIsSafe:
                    r += 0.35 * (prevDist - d)
                elif d < prevDist:
                    r -= 0.15
                r += 0.20 if s.apples < 3 else 0.03
                loopPressure = s.loopPressure()
                if loopPressure > 0.45:
                    r -= 2.0 * loopPressure
                prevDist = d

            tailOk, free, margin = s.survivalScore(s.body)
            r += 0.05 * free
            r += 0.5 if tailOk else -2.0
            if margin < 0:
                r += 0.8 * margin
            if len(s.body) > 8 and free < len(s.body) * 1.5:
                r -= 8.0

        r += 1200.0 * (s.apples ** 2) + 0.5 * len(uniqueCells)
        total += r

    g.fit = total / len(seeds)
    return g.fit


class Species:
    def __init__(self, rep):
        self.rep = rep
        self.members = []
        self.best = -float("inf")
        self.stagnation = 0

    def update(self):
        if not self.members:
            return
        top = max(m.fit for m in self.members)
        if top > self.best:
            self.best = top
            self.stagnation = 0
        else:
            self.stagnation += 1
        self.rep = random.choice(self.members)


class Population:
    def __init__(self):
        self.genomes = [seedGenome() for _ in range(POP_SIZE)]
        self.species = []
        self.gen = 0
        self.best = None

    def step(self, pool):
        global compat

        for s in self.species:
            s.members = []
        for g in self.genomes:
            for s in self.species:
                if g.dist(s.rep) < compat:
                    s.members.append(g)
                    break
            else:
                new = Species(g)
                new.members.append(g)
                self.species.append(new)
        self.species = [s for s in self.species if s.members]

        if len(self.species) > TARGET_SPECIES:
            compat += COMPAT_STEP
        elif len(self.species) < TARGET_SPECIES:
            compat = max(0.35, compat - COMPAT_STEP)

        start = (self.gen * EPISODES) % len(seedBank)
        seeds = [seedBank[(start + i) % len(seedBank)] for i in range(EPISODES)]
        jobs = [(g, seeds) for g in self.genomes]

        if pool:
            for g, f in zip(self.genomes, pool.map(evalGenome, jobs)):
                g.fit = f
        else:
            for job in jobs:
                evalGenome(job)

        top = max(self.genomes, key=lambda g: g.fit)
        if self.best is None or top.fit > self.best.fit:
            self.best = top.copy()

        minFit = min(g.fit for g in self.genomes)
        for s in self.species:
            s.update()
            for m in s.members:
                m.adj = (m.fit - minFit + 1e-6) / len(s.members)

        generationBest = top.copy()
        self.reproduce()
        return generationBest

    def reproduce(self):
        self.species.sort(key=lambda s: s.best, reverse=True)
        survivors = [s for s in self.species if s.stagnation < STAGNATION]
        self.species = survivors or self.species[:2]

        scores = [sum(max(0.0, m.adj) for m in s.members) for s in self.species]
        total = sum(scores)
        if total <= 0.0:
            scores = [1.0] * len(self.species)
            total = sum(scores)
        raw = [score / total * POP_SIZE for score in scores]
        quotas = [max(1, int(x)) for x in raw]

        while sum(quotas) > POP_SIZE:
            over = [i for i, q in enumerate(quotas) if q > 1]
            if not over:
                break
            quotas[max(over, key=lambda j: quotas[j])] -= 1
        while sum(quotas) < POP_SIZE:
            remainders = [raw[i] - int(raw[i]) for i in range(len(raw))]
            i = max(range(len(quotas)), key=lambda j: remainders[j])
            quotas[i] += 1
            raw[i] = int(raw[i])

        offspring = []
        for s, n in zip(self.species, quotas):
            s.members.sort(key=lambda g: g.fit, reverse=True)
            parents = s.members[:max(1, int(len(s.members) * SURVIVAL_RATE))]
            if len(s.members) >= 5:
                champ = parents[0].copy()
                champ.fit = champ.adj = 0.0
                offspring.append(champ)
                n -= 1
            for _ in range(n):
                offspring.append(makeChild(parents))

        weights = [max(1e-9, sum(max(0.0, m.adj) for m in s.members)) for s in self.species]
        while len(offspring) < POP_SIZE:
            s = random.choices(self.species, weights=weights, k=1)[0]
            s.members.sort(key=lambda g: g.fit, reverse=True)
            offspring.append(makeChild(s.members[:max(1, int(len(s.members) * SURVIVAL_RATE))]))

        offspring = offspring[:POP_SIZE]
        if self.best is not None and offspring:
            offspring[-1] = self.best.copy()

        self.genomes = offspring
        self.gen += 1


def makeChild(parents):
    if len(parents) > 1 and random.random() < CROSSOVER_RATE:
        child = cross(*random.sample(parents, 2))
    else:
        child = random.choice(parents).copy()
    child.mutate()
    child.fit = child.adj = 0.0
    return child
